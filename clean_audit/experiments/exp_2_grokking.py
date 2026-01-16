"""
EXPERIMENT 2 REVISED: Temporal Ordering via Grokking

Research Question: Does G lock before S during training?

Design: Use modular arithmetic with weight decay (known grokking setup)
- Grokking shows distinct phases: memorization → generalization
- Track G (QK drift) and S (amplitude, margin) metrics throughout
- Expected: A should be high during memorization, drop at grokking point

Success Criteria:
- Clear phase transition in accuracy (memorization → generalization)
- A_learned drops AFTER accuracy rises (G→S ordering)
- QK parameters stabilize before S redistributes
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List
import argparse
import json

from lib.logging_utils import setup_reproducibility
from lib.part_b_utils import get_qk_parameters, measure_suppressor_strength


class ModularAdditionDataset(Dataset):
    """Modular addition: (a + b) mod p"""
    
    def __init__(self, p: int = 113, split: str = 'train', train_frac: float = 0.3):
        self.p = p
        
        # Generate all pairs
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


class GrokTransformer(nn.Module):
    """Simple transformer for grokking experiments."""
    
    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.embed = nn.Embedding(p, d_model)
        self.pos_embed = nn.Embedding(2, d_model)
        
        # Single attention layer
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)
        
        self.cache = {}
    
    def forward(self, x):
        B, L = x.shape
        
        # Embeddings
        tok_emb = self.embed(x)
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)
        h = tok_emb + pos_emb
        
        # Attention
        h_ln = self.ln1(h)
        attn_out, attn_weights = self.attn(h_ln, h_ln, h_ln)
        h = h + attn_out
        self.cache['attn_weights'] = attn_weights.detach()
        
        # FFN
        h_ln = self.ln2(h)
        h = h + self.ffn(h_ln)
        
        # Output (use last position)
        h_final = self.ln_final(h[:, -1, :])
        self.cache['resid_post'] = h_final.detach()
        
        logits = self.head(h_final)
        return logits


def run_grokking_experiment(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    n_steps: int = 50000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    seed: int = 42,
    device: str = 'cuda'
):
    """
    Run grokking experiment with temporal tracking.
    
    Key insight: Grokking shows A dropping at generalization onset.
    If A drops AFTER accuracy rises, this proves G→S ordering.
    """
    
    print("\n" + "="*80)
    print("EXPERIMENT 2 REVISED: GROKKING TEMPORAL ORDERING")
    print("="*80 + "\n")
    
    setup_reproducibility(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Modulus p={p}, Weight Decay={weight_decay}\n")
    
    # Data
    train_dataset = ModularAdditionDataset(p=p, split='train')
    val_dataset = ModularAdditionDataset(p=p, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = GrokTransformer(p=p, d_model=d_model, n_heads=n_heads)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Initial QK for drift tracking
    initial_qk = get_qk_parameters(model).clone()
    
    # Tracking
    history = {
        'steps': [],
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'qk_drift': [],
        'resid_norm': [],
        'margin': []
    }
    
    print("\n" + "-"*80)
    print("TRAINING WITH GROKKING SETUP")
    print("-"*80 + "\n")
    
    train_iter = iter(train_loader)
    
    for step in range(n_steps):
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
        loss.backward()
        optimizer.step()
        
        # Log every 500 steps
        if step % 500 == 0 or step == n_steps - 1:
            model.eval()
            
            # Train accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                train_acc = (preds == y).float().mean().item()
            
            # Validation accuracy
            val_correct = 0
            val_total = 0
            margins = []
            resid_norms = []
            
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vlogits = model(vx)
                    
                    vpreds = vlogits.argmax(dim=-1)
                    val_correct += (vpreds == vy).sum().item()
                    val_total += vy.size(0)
                    
                    # Margin
                    true_logits = vlogits.gather(1, vy.unsqueeze(1)).squeeze()
                    max_other = vlogits.scatter(1, vy.unsqueeze(1), float('-inf')).max(dim=1).values
                    margin = (true_logits - max_other).mean().item()
                    margins.append(margin)
                    
                    # Residual norm
                    if 'resid_post' in model.cache:
                        resid_norms.append(model.cache['resid_post'].norm(dim=-1).mean().item())
            
            val_acc = val_correct / val_total
            avg_margin = np.mean(margins)
            avg_resid_norm = np.mean(resid_norms) if resid_norms else 0.0
            
            # QK drift
            current_qk = get_qk_parameters(model)
            qk_drift = (current_qk - initial_qk).norm().item()
            
            # Log
            history['steps'].append(step)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(loss.item())
            history['qk_drift'].append(qk_drift)
            history['resid_norm'].append(avg_resid_norm)
            history['margin'].append(avg_margin)
            
            print(f"Step {step:5d} | Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
                  f"QK Drift: {qk_drift:.2f} | Margin: {avg_margin:.2f} | Resid: {avg_resid_norm:.2f}")
            
            # Check for grokking
            if step > 0 and val_acc > 0.95 and history['val_acc'][-2] < 0.5:
                print(f"\n*** GROKKING DETECTED at step {step}! ***\n")
    
    # Analysis
    print("\n" + "="*80)
    print("GROKKING ANALYSIS")
    print("="*80 + "\n")
    
    # Find grokking point (val_acc crosses 90%)
    grokking_step = None
    for i, (s, acc) in enumerate(zip(history['steps'], history['val_acc'])):
        if acc > 0.90 and (i == 0 or history['val_acc'][i-1] < 0.90):
            grokking_step = s
            break
    
    # Find QK stabilization point (drift rate < 1% per 1k steps)
    qk_stable_step = None
    for i in range(1, len(history['steps'])):
        if i >= 2:
            drift_rate = (history['qk_drift'][i] - history['qk_drift'][i-1]) / \
                        max(1, (history['steps'][i] - history['steps'][i-1]))
            if drift_rate < 0.01:
                qk_stable_step = history['steps'][i]
                break
    
    # Find margin drop point
    margin_drop_step = None
    max_margin = max(history['margin'][:len(history['margin'])//2 + 1])  # Max in first half
    for i, (s, m) in enumerate(zip(history['steps'], history['margin'])):
        if m < max_margin * 0.7:  # 30% drop
            margin_drop_step = s
            break
    
    print(f"Grokking step (val_acc > 90%): {grokking_step}")
    print(f"QK stabilization step: {qk_stable_step}")
    print(f"Margin drop step: {margin_drop_step}")
    
    if grokking_step and qk_stable_step:
        if qk_stable_step < grokking_step:
            print(f"\n✓ G stabilized BEFORE grokking: G→S ordering confirmed!")
            ordering = "G_before_S"
        else:
            print(f"\n? G stabilized AFTER grokking: ordering unclear")
            ordering = "unclear"
    else:
        ordering = "no_grokking"
        print("\n✗ No clear grokking detected")
    
    results = {
        'history': history,
        'analysis': {
            'grokking_step': grokking_step,
            'qk_stable_step': qk_stable_step,
            'margin_drop_step': margin_drop_step,
            'ordering': ordering,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1]
        }
    }
    
    # Save
    output_path = Path("clean_audit/data/exp_2_grokking_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=113)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick_test', action='store_true')
    
    args = parser.parse_args()
    
    n_steps = 5000 if args.quick_test else args.n_steps
    
    run_grokking_experiment(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_steps=n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device
    )


if __name__ == '__main__':
    main()
