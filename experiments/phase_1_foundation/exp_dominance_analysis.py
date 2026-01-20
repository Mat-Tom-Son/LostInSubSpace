"""
Experiment 6.4: Head Dominance at Bifurcation

Motivated by IOI, Hydra, and circuit-tracing literature:
- Recovery succeeds when backup pathways already carry signal
- Collapse happens when no clear winner exists or heads conflict

Key question: Do recoveries show pre-existing alternative head dominance,
while collapses show uniform/conflicting dominance?

Measurements:
- Per-head contribution to final logits
- Per-head attention entropy
- Relative dominance score = head_contribution / sum(contributions)
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
# MODEL WITH DOMINANCE HOOKS
# =============================================================================

class TransformerBlockWithDominance(nn.Module):
    """Transformer block that exposes per-head contributions."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Manual attention for per-head access
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

        # Storage for dominance metrics (populated during forward)
        self.last_attn_weights = None  # [batch, n_heads, seq, seq]
        self.last_head_outputs = None  # [batch, n_heads, seq, head_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h_ln = self.ln1(x)

        # QKV projections
        Q = self.q_proj(h_ln).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h_ln).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h_ln).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
        self.last_attn_weights = attn_weights.detach()  # [B, H, L, L]

        # Per-head outputs (before concatenation)
        head_outputs = attn_weights @ V  # [B, H, L, head_dim]
        self.last_head_outputs = head_outputs.detach()

        # Concatenate and project
        attn_out = head_outputs.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.out_proj(attn_out)

        x = x + attn_out
        h_ln = self.ln2(x)
        x = x + self.ffn(h_ln)
        return x


class ModularArithmeticTransformerWithDominance(nn.Module):
    """Transformer with per-head dominance measurement."""

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
            TransformerBlockWithDominance(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
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

    def get_head_contributions(self, layer_idx: int = 1) -> Dict[str, torch.Tensor]:
        """
        Get per-head contribution metrics for a specific layer.

        Returns dict with:
        - 'attention_entropy': [n_heads] entropy of attention weights
        - 'output_magnitude': [n_heads] L2 norm of each head's output
        - 'dominance_score': [n_heads] relative contribution (sums to 1)
        """
        block = self.blocks[layer_idx]

        if block.last_attn_weights is None:
            return None

        # Attention entropy per head (averaged over batch and positions)
        # attn_weights: [B, H, L, L]
        attn = block.last_attn_weights
        # Compute entropy: -sum(p * log(p))
        eps = 1e-10
        entropy = -torch.sum(attn * torch.log(attn + eps), dim=-1)  # [B, H, L]
        entropy_per_head = entropy.mean(dim=(0, 2))  # [H]

        # Output magnitude per head
        # head_outputs: [B, H, L, head_dim]
        head_out = block.last_head_outputs
        # Take last position, compute L2 norm
        last_pos_out = head_out[:, :, -1, :]  # [B, H, head_dim]
        magnitude_per_head = last_pos_out.norm(dim=-1).mean(dim=0)  # [H]

        # Dominance score (relative contribution)
        total_magnitude = magnitude_per_head.sum()
        dominance_score = magnitude_per_head / (total_magnitude + 1e-10)

        return {
            'attention_entropy': entropy_per_head,
            'output_magnitude': magnitude_per_head,
            'dominance_score': dominance_score
        }


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
    """Reinitialize Q and K matrices for the first N attention heads."""
    if n_heads_to_reinit == 0:
        return

    block = model.blocks[layer_idx]
    d_model = model.d_model
    n_heads = model.n_heads
    head_dim = d_model // n_heads

    with torch.no_grad():
        for head_idx in range(n_heads_to_reinit):
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim

            # Reinit Q projection for this head
            nn.init.xavier_uniform_(block.q_proj.weight[start:end, :])
            if block.q_proj.bias is not None:
                nn.init.zeros_(block.q_proj.bias[start:end])

            # Reinit K projection for this head
            nn.init.xavier_uniform_(block.k_proj.weight[start:end, :])
            if block.k_proj.bias is not None:
                nn.init.zeros_(block.k_proj.bias[start:end])


# =============================================================================
# VALIDATION WITH DOMINANCE
# =============================================================================

def validate_with_dominance(model: nn.Module, dataloader: DataLoader, device: str,
                            intervention_layer: int = 1) -> Dict:
    """Validate and compute dominance metrics."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    # Accumulate dominance metrics
    all_entropy = []
    all_magnitude = []
    all_dominance = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)

            loss = F.cross_entropy(logits, targets)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_loss += loss.item() * inputs.shape[0]
            total_samples += inputs.shape[0]

            # Get dominance metrics
            dom = model.get_head_contributions(layer_idx=intervention_layer)
            if dom is not None:
                all_entropy.append(dom['attention_entropy'].cpu())
                all_magnitude.append(dom['output_magnitude'].cpu())
                all_dominance.append(dom['dominance_score'].cpu())

    # Average dominance metrics
    metrics = {
        'accuracy': total_correct / total_samples,
        'loss': total_loss / total_samples,
    }

    if all_entropy:
        metrics['head_entropy'] = torch.stack(all_entropy).mean(dim=0).tolist()
        metrics['head_magnitude'] = torch.stack(all_magnitude).mean(dim=0).tolist()
        metrics['head_dominance'] = torch.stack(all_dominance).mean(dim=0).tolist()

        # Summary stats
        dom_scores = torch.tensor(metrics['head_dominance'])
        metrics['dominance_max'] = dom_scores.max().item()
        metrics['dominance_std'] = dom_scores.std().item()
        metrics['dominance_gini'] = compute_gini(dom_scores).item()

    return metrics


def compute_gini(x: torch.Tensor) -> torch.Tensor:
    """Compute Gini coefficient (0 = equal, 1 = one dominates)."""
    x = x.sort().values
    n = len(x)
    cumsum = x.cumsum(0)
    return (2 * torch.arange(1, n+1, device=x.device) * x - cumsum[-1]).sum() / (n * cumsum[-1] + 1e-10) - (n + 1) / n


# =============================================================================
# TRAINING
# =============================================================================

def train_baseline(model, train_loader, val_loader, device, lr=1e-3, weight_decay=1.0,
                   max_steps=50000, target_accuracy=0.99, log_interval=500,
                   intervention_layer=1):
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
            metrics = validate_with_dominance(model, val_loader, device, intervention_layer)
            metrics['step'] = step + 1
            trajectory.append(metrics)
            pbar.set_postfix({'val_acc': f"{metrics['accuracy']*100:.1f}%"})
            if metrics['accuracy'] >= target_accuracy:
                print(f"\n[OK] Converged at step {step+1}")
                return metrics, trajectory

    return validate_with_dominance(model, val_loader, device, intervention_layer), trajectory


def train_recovery_with_dominance(model, train_loader, val_loader, device,
                                   lr=1e-3, weight_decay=1.0, n_steps=20000,
                                   log_interval=100, intervention_layer=1,
                                   desc="Recovery"):
    """Recovery training with dominance tracking."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)
    trajectory = []

    # Log initial state
    metrics = validate_with_dominance(model, val_loader, device, intervention_layer)
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
            metrics = validate_with_dominance(model, val_loader, device, intervention_layer)
            metrics['step'] = step + 1
            trajectory.append(metrics)

            dom_max = metrics.get('dominance_max', 0)
            pbar.set_postfix({
                'acc': f"{metrics['accuracy']*100:.1f}%",
                'dom_max': f"{dom_max:.2f}"
            })

    return trajectory


# =============================================================================
# TRAJECTORY CLASSIFICATION
# =============================================================================

def find_bifurcation_point(trajectory: List[Dict], baseline_acc: float = 1.0) -> Dict:
    """
    Find the bifurcation point (first collapse or recovery event).

    Returns:
    - 'type': 'collapse' or 'recovery' or None
    - 'step_idx': index in trajectory
    - 'step': actual step number
    """
    accuracies = [t['accuracy'] for t in trajectory]

    # Find first major drop (>30% from recent max)
    for i in range(5, len(accuracies)):
        recent_max = max(accuracies[max(0, i-5):i])
        if recent_max - accuracies[i] > 0.30:
            return {'type': 'collapse', 'step_idx': i, 'step': trajectory[i]['step']}

    # Find first major recovery (>30% rise from recent min)
    for i in range(5, len(accuracies)):
        recent_min = min(accuracies[max(0, i-5):i])
        if accuracies[i] - recent_min > 0.30 and accuracies[i] > 0.9:
            return {'type': 'recovery', 'step_idx': i, 'step': trajectory[i]['step']}

    return {'type': None, 'step_idx': None, 'step': None}


def classify_trajectory(trajectory: List[Dict], baseline_acc: float = 1.0) -> str:
    """Classify trajectory into STABLE, COLLAPSE_THEN_RECOVER, IRREVERSIBLE_COLLAPSE, PARTIAL_RECOVERY."""
    accuracies = [t['accuracy'] for t in trajectory]
    final_acc = accuracies[-1]

    # Check for collapses
    has_collapse = False
    for i in range(5, len(accuracies)):
        recent_max = max(accuracies[max(0, i-5):i])
        if recent_max - accuracies[i] > 0.30:
            has_collapse = True
            break

    recovered = final_acc > 0.9 * baseline_acc
    collapsed = final_acc < 0.5 * baseline_acc

    if recovered and not has_collapse:
        return "STABLE"
    elif recovered and has_collapse:
        return "COLLAPSE_THEN_RECOVER"
    elif collapsed:
        return "IRREVERSIBLE_COLLAPSE"
    else:
        return "PARTIAL_RECOVERY"


def analyze_pre_bifurcation_dominance(trajectory: List[Dict], bifurcation_idx: int,
                                       window_size: int = 5) -> Dict:
    """
    Analyze dominance patterns in the window before bifurcation.

    Returns summary statistics of head dominance before the event.
    """
    if bifurcation_idx is None or bifurcation_idx < window_size:
        return None

    start_idx = max(0, bifurcation_idx - window_size)
    end_idx = bifurcation_idx

    window = trajectory[start_idx:end_idx]

    # Aggregate dominance metrics
    dominances = [t['head_dominance'] for t in window if 'head_dominance' in t]
    if not dominances:
        return None

    dominances = np.array(dominances)  # [window, n_heads]

    return {
        'mean_dominance': dominances.mean(axis=0).tolist(),
        'std_dominance': dominances.std(axis=0).tolist(),
        'max_head': int(dominances.mean(axis=0).argmax()),
        'max_dominance': float(dominances.mean(axis=0).max()),
        'dominance_spread': float(dominances.mean(axis=0).std()),
        'gini_mean': float(np.mean([compute_gini(torch.tensor(d)).item() for d in dominances]))
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_dominance_analysis(
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
    seeds: List[int] = None,
    intervention_layer: int = 1,
    heads_to_test: List[int] = None,
    quick_test: bool = False
):
    if seeds is None:
        seeds = [42, 43, 44, 45, 46, 47]
    if heads_to_test is None:
        heads_to_test = [1, 2, 3]

    if quick_test:
        baseline_max_steps = 5000
        recovery_steps = 3000
        target_accuracy = 0.95
        seeds = [42, 43]
        print("\n*** QUICK TEST MODE ***\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print("EXPERIMENT 6.4: Head Dominance at Bifurcation")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Heads to test: {heads_to_test}")
    print(f"Measuring: per-head contribution, entropy, dominance score")
    print()

    all_results = {}
    bifurcation_analyses = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        setup_reproducibility(seed)
        all_results[seed] = {}

        # Data
        train_dataset = ModularAdditionDataset(p=p, is_train=True, seed=seed)
        val_dataset = ModularAdditionDataset(p=p, is_train=False, seed=seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Baseline training
        print("\n--- Baseline Training ---")
        model = ModularArithmeticTransformerWithDominance(
            p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff
        )
        baseline_metrics, _ = train_baseline(
            model, train_loader, val_loader, device,
            lr=lr, weight_decay=weight_decay,
            max_steps=baseline_max_steps, target_accuracy=target_accuracy,
            intervention_layer=intervention_layer
        )

        checkpoint = copy.deepcopy(model.state_dict())
        baseline_acc = baseline_metrics['accuracy']
        print(f"Baseline accuracy: {baseline_acc*100:.2f}%")

        all_results[seed]['baseline_acc'] = baseline_acc
        all_results[seed]['conditions'] = {}

        # Test each head count
        for n_heads_reinit in heads_to_test:
            print(f"\n--- {n_heads_reinit} heads reinitialized ---")

            # Restore from checkpoint
            model.load_state_dict(copy.deepcopy(checkpoint))
            model.to(device)

            # Apply intervention
            reinit_qk_n_heads(model, layer_idx=intervention_layer, n_heads_to_reinit=n_heads_reinit)

            # Recovery training with dominance tracking
            trajectory = train_recovery_with_dominance(
                model, train_loader, val_loader, device,
                lr=lr, weight_decay=weight_decay,
                n_steps=recovery_steps,
                log_interval=100,
                intervention_layer=intervention_layer,
                desc=f"S{seed}-{n_heads_reinit}H"
            )

            # Classify trajectory
            category = classify_trajectory(trajectory, baseline_acc)

            # Find bifurcation point
            bifurcation = find_bifurcation_point(trajectory, baseline_acc)

            # Analyze pre-bifurcation dominance
            pre_bif_dominance = None
            if bifurcation['step_idx'] is not None:
                pre_bif_dominance = analyze_pre_bifurcation_dominance(
                    trajectory, bifurcation['step_idx'], window_size=5
                )

            print(f"  Category: {category}")
            print(f"  Final: {trajectory[-1]['accuracy']*100:.1f}%")
            if bifurcation['type']:
                print(f"  Bifurcation: {bifurcation['type']} at step {bifurcation['step']}")
            if pre_bif_dominance:
                print(f"  Pre-bif dominance max: {pre_bif_dominance['max_dominance']:.3f} (head {pre_bif_dominance['max_head']})")
                print(f"  Pre-bif Gini: {pre_bif_dominance['gini_mean']:.3f}")

            all_results[seed]['conditions'][n_heads_reinit] = {
                'trajectory': trajectory,
                'category': category,
                'bifurcation': bifurcation,
                'pre_bifurcation_dominance': pre_bif_dominance
            }

            bifurcation_analyses.append({
                'seed': seed,
                'n_heads': n_heads_reinit,
                'category': category,
                'final_acc': trajectory[-1]['accuracy'],
                'bifurcation_type': bifurcation['type'],
                'bifurcation_step': bifurcation['step'],
                'pre_bif_dominance': pre_bif_dominance
            })

    # ==========================================================================
    # AGGREGATE ANALYSIS: RECOVERY vs COLLAPSE DOMINANCE PATTERNS
    # ==========================================================================

    print("\n" + "=" * 70)
    print("DOMINANCE ANALYSIS: RECOVERY vs COLLAPSE")
    print("=" * 70)

    # Separate by outcome
    recoveries = [b for b in bifurcation_analyses if b['category'] == 'COLLAPSE_THEN_RECOVER']
    collapses = [b for b in bifurcation_analyses if b['category'] == 'IRREVERSIBLE_COLLAPSE']

    print(f"\nN recoveries: {len(recoveries)}")
    print(f"N irreversible collapses: {len(collapses)}")

    # Analyze pre-bifurcation dominance patterns
    print("\n--- Pre-Bifurcation Dominance Comparison ---")

    if recoveries:
        recovery_ginis = [r['pre_bif_dominance']['gini_mean']
                         for r in recoveries if r['pre_bif_dominance']]
        recovery_max_doms = [r['pre_bif_dominance']['max_dominance']
                            for r in recoveries if r['pre_bif_dominance']]
        if recovery_ginis:
            print(f"\nRECOVERIES (n={len(recovery_ginis)}):")
            print(f"  Gini (inequality): {np.mean(recovery_ginis):.3f} +/- {np.std(recovery_ginis):.3f}")
            print(f"  Max dominance:     {np.mean(recovery_max_doms):.3f} +/- {np.std(recovery_max_doms):.3f}")
            print(f"  -> Higher Gini = more unequal = one head dominates")

    if collapses:
        collapse_ginis = [c['pre_bif_dominance']['gini_mean']
                         for c in collapses if c['pre_bif_dominance']]
        collapse_max_doms = [c['pre_bif_dominance']['max_dominance']
                            for c in collapses if c['pre_bif_dominance']]
        if collapse_ginis:
            print(f"\nCOLLAPSES (n={len(collapse_ginis)}):")
            print(f"  Gini (inequality): {np.mean(collapse_ginis):.3f} +/- {np.std(collapse_ginis):.3f}")
            print(f"  Max dominance:     {np.mean(collapse_max_doms):.3f} +/- {np.std(collapse_max_doms):.3f}")
            print(f"  -> Lower Gini = more equal = no clear winner")

    # Statistical comparison
    if recovery_ginis and collapse_ginis:
        print("\n--- Statistical Comparison ---")
        gini_diff = np.mean(recovery_ginis) - np.mean(collapse_ginis)
        dom_diff = np.mean(recovery_max_doms) - np.mean(collapse_max_doms)

        print(f"Gini difference (recovery - collapse): {gini_diff:+.3f}")
        print(f"Max dominance difference:              {dom_diff:+.3f}")

        if gini_diff > 0.05:
            print("\n[FINDING] Recoveries show HIGHER head inequality (one head dominates)")
            print("          This matches IOI/Hydra: backup pathway already carries signal")
        elif gini_diff < -0.05:
            print("\n[FINDING] Collapses show HIGHER head inequality")
            print("          This contradicts the expected pattern - investigate further")
        else:
            print("\n[FINDING] No clear dominance difference between outcomes")
            print("          Bifurcation may depend on factors other than head dominance")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = {
        'config': {
            'p': p, 'd_model': d_model, 'n_heads': n_heads, 'n_layers': n_layers,
            'd_ff': d_ff, 'lr': lr, 'weight_decay': weight_decay,
            'recovery_steps': recovery_steps, 'seeds': seeds,
            'heads_to_test': heads_to_test, 'intervention_layer': intervention_layer
        },
        'bifurcation_analyses': bifurcation_analyses,
        'summary': {
            'n_recoveries': len(recoveries),
            'n_collapses': len(collapses),
            'recovery_gini_mean': float(np.mean(recovery_ginis)) if recovery_ginis else None,
            'recovery_gini_std': float(np.std(recovery_ginis)) if recovery_ginis else None,
            'collapse_gini_mean': float(np.mean(collapse_ginis)) if collapse_ginis else None,
            'collapse_gini_std': float(np.std(collapse_ginis)) if collapse_ginis else None,
        },
        'run_id': run_id
    }

    json_path = output_dir / f"exp_dominance_analysis_{run_id}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {json_path}")

    # Full trajectories
    full_path = output_dir / f"exp_dominance_analysis_{run_id}_full.json"
    with open(full_path, 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, default=str)
    print(f"Full trajectories saved to: {full_path}")

    # Plot
    plot_path = output_dir / f"exp_dominance_analysis_{run_id}.png"
    plot_dominance_analysis(bifurcation_analyses, str(plot_path))

    return summary, all_results


def plot_dominance_analysis(analyses: List[Dict], save_path: str):
    """Visualize dominance patterns for recovery vs collapse."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    recoveries = [a for a in analyses if a['category'] == 'COLLAPSE_THEN_RECOVER']
    collapses = [a for a in analyses if a['category'] == 'IRREVERSIBLE_COLLAPSE']

    # Plot 1: Gini comparison
    ax1 = axes[0, 0]
    recovery_ginis = [r['pre_bif_dominance']['gini_mean']
                     for r in recoveries if r['pre_bif_dominance']]
    collapse_ginis = [c['pre_bif_dominance']['gini_mean']
                     for c in collapses if c['pre_bif_dominance']]

    if recovery_ginis or collapse_ginis:
        positions = []
        data = []
        labels = []
        if recovery_ginis:
            positions.append(1)
            data.append(recovery_ginis)
            labels.append(f'Recovery\n(n={len(recovery_ginis)})')
        if collapse_ginis:
            positions.append(2)
            data.append(collapse_ginis)
            labels.append(f'Collapse\n(n={len(collapse_ginis)})')

        bp = ax1.boxplot(data, positions=positions, widths=0.6)
        ax1.set_xticks(positions)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Gini Coefficient')
        ax1.set_title('Head Dominance Inequality\n(Higher = One Head Dominates)')
        ax1.grid(True, alpha=0.3)

    # Plot 2: Max dominance comparison
    ax2 = axes[0, 1]
    recovery_max = [r['pre_bif_dominance']['max_dominance']
                   for r in recoveries if r['pre_bif_dominance']]
    collapse_max = [c['pre_bif_dominance']['max_dominance']
                   for c in collapses if c['pre_bif_dominance']]

    if recovery_max or collapse_max:
        positions = []
        data = []
        labels = []
        if recovery_max:
            positions.append(1)
            data.append(recovery_max)
            labels.append(f'Recovery\n(n={len(recovery_max)})')
        if collapse_max:
            positions.append(2)
            data.append(collapse_max)
            labels.append(f'Collapse\n(n={len(collapse_max)})')

        bp = ax2.boxplot(data, positions=positions, widths=0.6)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Max Head Dominance')
        ax2.set_title('Strongest Head Contribution\n(Before Bifurcation)')
        ax2.grid(True, alpha=0.3)

    # Plot 3: Scatter of Gini vs final accuracy
    ax3 = axes[1, 0]
    for a in analyses:
        if a['pre_bif_dominance']:
            color = 'green' if a['category'] == 'COLLAPSE_THEN_RECOVER' else 'red'
            marker = 'o' if a['category'] == 'COLLAPSE_THEN_RECOVER' else 'x'
            ax3.scatter(a['pre_bif_dominance']['gini_mean'], a['final_acc'],
                       c=color, marker=marker, s=100, alpha=0.7)

    ax3.scatter([], [], c='green', marker='o', label='Recovery')
    ax3.scatter([], [], c='red', marker='x', label='Collapse')
    ax3.set_xlabel('Pre-Bifurcation Gini')
    ax3.set_ylabel('Final Accuracy')
    ax3.set_title('Dominance Inequality vs Outcome')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Per-head dominance distribution
    ax4 = axes[1, 1]

    # Aggregate head dominances
    recovery_heads = []
    collapse_heads = []
    for a in analyses:
        if a['pre_bif_dominance'] and 'mean_dominance' in a['pre_bif_dominance']:
            if a['category'] == 'COLLAPSE_THEN_RECOVER':
                recovery_heads.append(a['pre_bif_dominance']['mean_dominance'])
            elif a['category'] == 'IRREVERSIBLE_COLLAPSE':
                collapse_heads.append(a['pre_bif_dominance']['mean_dominance'])

    if recovery_heads:
        recovery_avg = np.mean(recovery_heads, axis=0)
        ax4.bar(np.arange(len(recovery_avg)) - 0.2, recovery_avg, 0.4,
               label='Recovery', color='green', alpha=0.7)
    if collapse_heads:
        collapse_avg = np.mean(collapse_heads, axis=0)
        ax4.bar(np.arange(len(collapse_avg)) + 0.2, collapse_avg, 0.4,
               label='Collapse', color='red', alpha=0.7)

    ax4.set_xlabel('Head Index')
    ax4.set_ylabel('Mean Dominance Score')
    ax4.set_title('Per-Head Contribution\n(Before Bifurcation)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Head Dominance Analysis")
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
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46, 47])
    parser.add_argument('--heads', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--intervention_layer', type=int, default=1)
    parser.add_argument('--quick_test', action='store_true')

    args = parser.parse_args()

    run_dominance_analysis(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        baseline_max_steps=args.baseline_max_steps,
        recovery_steps=args.recovery_steps,
        target_accuracy=args.target_accuracy,
        seeds=args.seeds,
        intervention_layer=args.intervention_layer,
        heads_to_test=args.heads,
        quick_test=args.quick_test
    )


if __name__ == '__main__':
    main()
