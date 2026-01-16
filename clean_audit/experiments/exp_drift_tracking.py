
"""
EXPERIMENT 1.2: QK/V/MLP Drift Tracking

Goal: Quantify stabilization (freezing) rates of different parameter groups.
Hypothesis: Geometry Annealing implies QK freezes BEFORE OV/MLP.

Method:
1. Train Modular Arithmetic model with Weight Decay (Grokking setup)
2. Track "Velocity" (norm of change per step) for QK, OV, MLP
3. Compare velocity curves
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt

from lib.logging_utils import setup_reproducibility
from lib.part_b_utils import get_qk_parameters, get_ov_parameters, get_mlp_parameters


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


class CausalAttention(nn.Module):
    """
    Causal Attention with EXPLICITLY SEPARATE Q, K, V projections.
    This enables precise tracking of QK (Geometry) vs V (Slack) drift.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Explicit separate layers
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        B, L, D = x.shape
        
        # Projections
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        
        # Causal mask
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class GrokTransformer(nn.Module):
    """Transformer for drift tracking."""
    
    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.p = p
        
        self.embed = nn.Embedding(p, d_model)
        self.pos_embed = nn.Embedding(2, d_model)
        
        # Block 1
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)
    
    def forward(self, x):
        B, L = x.shape
        
        # Embeddings
        tok_emb = self.embed(x)
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = tok_emb + self.pos_embed(pos)
        
        # Attention
        h = h + self.attn(self.ln1(h))
        
        # FFN
        h = h + self.ffn(self.ln2(h))
        
        # Output
        h_final = self.ln_final(h[:, -1, :])
        logits = self.head(h_final)
        return logits


def run_drift_experiment(
    p: int = 113,
    n_steps: int = 20000,
    seed: int = 42,
    device: str = 'cuda',
    quick_test: bool = False
):
    print(f"\nrunning drift tracking experiment (seed={seed})...")
    setup_reproducibility(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    train_dataset = ModularAdditionDataset(p=p, split='train', train_frac=0.3)
    val_dataset = ModularAdditionDataset(p=p, split='val', train_frac=0.3)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # Model
    model = GrokTransformer(p=p, d_model=128, n_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    
    # Tracking
    history = {
        'steps': [],
        'val_acc': [],
        'vel_qk': [],
        'vel_ov': [],
        'vel_mlp': [],
        'disp_qk': [],
        'disp_ov': [],
        'disp_mlp': []
    }
    
    # Initial params
    init_qk = get_qk_parameters(model).clone()
    init_ov = get_ov_parameters(model).clone()
    init_mlp = get_mlp_parameters(model).clone()
    
    prev_qk = init_qk.clone()
    prev_ov = init_ov.clone()
    prev_mlp = init_mlp.clone()
    
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
        
        # Log 
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                # Val Acc
                correct = 0
                total = 0
                for vx, vy in val_loader:
                    vlogits = model(vx.to(device))
                    correct += (vlogits.argmax(-1) == vy.to(device)).sum().item()
                    total += vy.size(0)
                val_acc = correct / total
                
                # Parameters
                curr_qk = get_qk_parameters(model)
                curr_ov = get_ov_parameters(model)
                curr_mlp = get_mlp_parameters(model)
                
                # Velocity: norm(curr - prev) / norm(init) (Relative velocity?)
                # Or just raw norm? Raw norm is scale dependent.
                # Let's normalize by parameter count or init norm.
                # Relative Velocity: ||curr - prev|| / ||curr|| 
                # Or just ||curr - prev||.
                # Let's use ||curr - prev||.
                
                v_qk = (curr_qk - prev_qk).norm().item()
                v_ov = (curr_ov - prev_ov).norm().item()
                v_mlp = (curr_mlp - prev_mlp).norm().item()
                
                # Displacement: ||curr - init|| / ||init||
                d_qk = (curr_qk - init_qk).norm().item() / (init_qk.norm().item() + 1e-9)
                d_ov = (curr_ov - init_ov).norm().item() / (init_ov.norm().item() + 1e-9)
                d_mlp = (curr_mlp - init_mlp).norm().item() / (init_mlp.norm().item() + 1e-9)
                
                history['steps'].append(step)
                history['val_acc'].append(val_acc)
                history['vel_qk'].append(v_qk)
                history['vel_ov'].append(v_ov)
                history['vel_mlp'].append(v_mlp)
                history['disp_qk'].append(d_qk)
                history['disp_ov'].append(d_ov)
                history['disp_mlp'].append(d_mlp)
                
                # Update prev
                prev_qk = curr_qk.clone()
                prev_ov = curr_ov.clone()
                prev_mlp = curr_mlp.clone()
                
                if step % 1000 == 0:
                    print(f"Step {step}: Val Acc={val_acc:.2%}, Vel QK={v_qk:.4f}, OV={v_ov:.4f}, MLP={v_mlp:.4f}")
    
    return history

def plot_drift(history, output_path='paper/drift_tracking.png', max_step=None):
    steps = np.array(history['steps'])
    
    # Crop if requested
    if max_step is not None:
        mask = steps <= max_step
        steps = steps[mask]
        # Crop all lists in history that same way
        # But history is dict of lists.
        # We need to slice them.
        cropped_history = {}
        for k, v in history.items():
            if k == 'steps': continue
            cropped_history[k] = np.array(v)[mask]
        
        history = cropped_history
        history['steps'] = steps
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # 1. Accuracy
    ax1.plot(steps, history['val_acc'], color='black', label='Val Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Grokking Progress')
    ax1.grid(True, alpha=0.3)
    
    # 2. Velocity (Smoothed)
    # Simple moving average
    def smooth(y, box_pts=10):
        # Handle short arrays
        if len(y) < box_pts: return y
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        # Fix edges
        y_smooth[:box_pts] = y[:box_pts]
        y_smooth[-box_pts:] = y[-box_pts:]
        return y_smooth
    
    ax2.plot(steps, smooth(history['vel_qk']), label='QK (Geometry)', color='blue', linewidth=2)
    ax2.plot(steps, smooth(history['vel_ov']), label='OV (Slack)', color='orange', linewidth=2)
    ax2.plot(steps, smooth(history['vel_mlp']), label='MLP (Slack)', color='green', linewidth=2)
    ax2.set_ylabel('Velocity (||θ_t - θ_{t-1}||)')
    ax2.set_title('Parameter Update Velocity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Displacement
    ax3.plot(steps, history['disp_qk'], label='QK', color='blue', linewidth=2)
    ax3.plot(steps, history['disp_ov'], label='OV', color='orange', linewidth=2)
    ax3.plot(steps, history['disp_mlp'], label='MLP', color='green', linewidth=2)
    ax3.set_ylabel('Displacement (||Δθ|| / ||θ_0||)')
    ax3.set_title('Total Parameter Drift')
    ax3.set_xlabel('Steps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick_test', action='store_true')
    parser.add_argument('--output_dir', type=str, default='clean_audit/data')
    parser.add_argument('--load_results', type=str, default=None, help='Load JSON results file')
    parser.add_argument('--plot_max_step', type=int, default=None, help='Limit plot x-axis to N steps')
    args = parser.parse_args()
    
    if args.load_results:
        print(f"Loading results from {args.load_results}")
        with open(args.load_results, 'r') as f:
            history = json.load(f)
    else:
        n_steps = 1000 if args.quick_test else 25000
        history = run_drift_experiment(n_steps=n_steps, quick_test=args.quick_test)
        
        # Save JSON
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'exp_drift_results.json', 'w') as f:
            json.dump(history, f, indent=2)
        
    plot_drift(history, max_step=args.plot_max_step)

if __name__ == '__main__':
    main()
