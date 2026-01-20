"""
Experiment 6.3: Time-to-Collapse Trajectory Analysis

Same setup as head sweep, but with:
- Denser logging (every 100 steps)
- Additional diagnostic metrics (loss variance, margin)
- Trajectory classification (STABLE, COLLAPSE_THEN_RECOVER, IRREVERSIBLE_COLLAPSE)
- Focus on partial damage zone (1, 2, 3 heads)

Goal: Transform "chaotic variance" into trajectory classes.
Identify whether collapses are fundamentally chaotic or have hidden structure.
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
from collections import deque

from lib.logging_utils import setup_reproducibility


# =============================================================================
# MODEL (same as head_sweep)
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
    """Reinitialize Q and K matrices for the first N attention heads."""
    if n_heads_to_reinit == 0:
        return

    block = model.blocks[layer_idx]
    attn = block.attn
    d_model = attn.embed_dim
    n_heads = attn.num_heads
    head_dim = d_model // n_heads

    with torch.no_grad():
        for head_idx in range(n_heads_to_reinit):
            start_row = head_idx * head_dim
            end_row = (head_idx + 1) * head_dim

            attn.in_proj_weight[start_row:end_row, :] = torch.nn.init.xavier_uniform_(
                torch.empty(head_dim, d_model, device=attn.in_proj_weight.device)
            )

            k_start = d_model + start_row
            k_end = d_model + end_row
            attn.in_proj_weight[k_start:k_end, :] = torch.nn.init.xavier_uniform_(
                torch.empty(head_dim, d_model, device=attn.in_proj_weight.device)
            )

            if attn.in_proj_bias is not None:
                nn.init.zeros_(attn.in_proj_bias[start_row:end_row])
                nn.init.zeros_(attn.in_proj_bias[k_start:k_end])


# =============================================================================
# VALIDATION WITH MARGIN
# =============================================================================

def validate_with_margin(model: nn.Module, dataloader: DataLoader, device: str) -> Dict[str, float]:
    """Validate and compute margin (logit gap between correct and second-highest)."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    total_margin = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)

            loss = F.cross_entropy(logits, targets)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_loss += loss.item() * inputs.shape[0]
            total_samples += inputs.shape[0]

            # Compute margin: logit[correct] - max(logit[other])
            batch_size = logits.shape[0]
            correct_logits = logits[torch.arange(batch_size), targets]

            # Mask out correct class to find second highest
            masked_logits = logits.clone()
            masked_logits[torch.arange(batch_size), targets] = float('-inf')
            second_highest = masked_logits.max(dim=-1).values

            margin = (correct_logits - second_highest).mean().item()
            total_margin += margin * inputs.shape[0]

    return {
        'accuracy': total_correct / total_samples,
        'loss': total_loss / total_samples,
        'margin': total_margin / total_samples
    }


# =============================================================================
# TRAINING
# =============================================================================

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
            metrics = validate_with_margin(model, val_loader, device)
            metrics['step'] = step + 1
            trajectory.append(metrics)
            pbar.set_postfix({'val_acc': f"{metrics['accuracy']*100:.1f}%"})
            if metrics['accuracy'] >= target_accuracy:
                print(f"\n[OK] Converged at step {step+1}")
                return metrics, trajectory

    return validate_with_margin(model, val_loader, device), trajectory


def train_recovery_dense(model, train_loader, val_loader, device, lr=1e-3, weight_decay=1.0,
                         n_steps=20000, log_interval=100, desc="Recovery"):
    """
    Recovery training with dense logging for trajectory analysis.

    Logs every 100 steps:
    - accuracy
    - loss
    - margin
    - loss_variance (rolling 5-step window)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)
    trajectory = []

    # Rolling window for loss variance
    loss_window = deque(maxlen=5)

    # Log initial state
    metrics = validate_with_margin(model, val_loader, device)
    metrics['step'] = 0
    metrics['loss_variance'] = 0.0
    trajectory.append(metrics)
    loss_window.append(metrics['loss'])

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
            metrics = validate_with_margin(model, val_loader, device)
            metrics['step'] = step + 1

            # Update loss window and compute variance
            loss_window.append(metrics['loss'])
            if len(loss_window) >= 2:
                metrics['loss_variance'] = np.var(list(loss_window))
            else:
                metrics['loss_variance'] = 0.0

            trajectory.append(metrics)
            pbar.set_postfix({
                'acc': f"{metrics['accuracy']*100:.1f}%",
                'margin': f"{metrics['margin']:.2f}"
            })

    return trajectory


# =============================================================================
# TRAJECTORY CLASSIFICATION
# =============================================================================

def find_sharp_drops(accuracies: List[float], threshold: float = 0.30, window: int = 5) -> List[int]:
    """Find indices where accuracy drops more than threshold within window steps."""
    drops = []
    for i in range(window, len(accuracies)):
        max_prev = max(accuracies[i-window:i])
        if max_prev - accuracies[i] > threshold:
            drops.append(i)
    return drops


def find_sharp_rises(accuracies: List[float], threshold: float = 0.30, window: int = 5) -> List[int]:
    """Find indices where accuracy rises more than threshold within window steps."""
    rises = []
    for i in range(window, len(accuracies)):
        min_prev = min(accuracies[i-window:i])
        if accuracies[i] - min_prev > threshold:
            rises.append(i)
    return rises


def classify_trajectory(trajectory: List[Dict], baseline_acc: float = 1.0) -> Dict:
    """
    Classify a trajectory into one of:
    - STABLE: Final acc > 90% baseline, no major collapses
    - COLLAPSE_THEN_RECOVER: Final acc > 90% baseline, had collapse(s)
    - IRREVERSIBLE_COLLAPSE: Final acc < 50% baseline
    - PARTIAL_RECOVERY: Everything else

    Returns classification and statistics.
    """
    accuracies = [t['accuracy'] for t in trajectory]
    steps = [t['step'] for t in trajectory]

    final_acc = accuracies[-1]

    # Find collapse events
    collapses = find_sharp_drops(accuracies, threshold=0.30, window=5)
    recoveries = find_sharp_rises(accuracies, threshold=0.30, window=5)

    # Find first sharp drop time
    first_collapse_step = steps[collapses[0]] if collapses else None

    # Track minimum accuracy and when it occurred
    min_acc = min(accuracies)
    min_acc_idx = accuracies.index(min_acc)
    min_acc_step = steps[min_acc_idx]

    # Classify
    recovered = final_acc > 0.9 * baseline_acc
    collapsed = final_acc < 0.5 * baseline_acc

    if recovered and len(collapses) == 0:
        category = "STABLE"
    elif recovered and len(collapses) > 0:
        category = "COLLAPSE_THEN_RECOVER"
    elif collapsed:
        category = "IRREVERSIBLE_COLLAPSE"
    else:
        category = "PARTIAL_RECOVERY"

    return {
        'category': category,
        'final_acc': final_acc,
        'n_collapses': len(collapses),
        'n_recoveries': len(recoveries),
        'first_collapse_step': first_collapse_step,
        'min_acc': min_acc,
        'min_acc_step': min_acc_step,
        'collapse_indices': collapses,
        'recovery_indices': recoveries
    }


def analyze_precursors(trajectory: List[Dict], collapse_indices: List[int],
                       window_before: int = 5) -> Dict:
    """
    Analyze whether there are consistent precursor signals before collapses.

    Looks at loss_variance and margin in the window before each collapse.
    """
    if not collapse_indices:
        return {'n_collapses_analyzed': 0}

    loss_var_before = []
    margin_before = []
    loss_var_baseline = []
    margin_baseline = []

    for idx in collapse_indices:
        if idx >= window_before:
            # Get values in window before collapse
            for i in range(idx - window_before, idx):
                if 'loss_variance' in trajectory[i]:
                    loss_var_before.append(trajectory[i]['loss_variance'])
                if 'margin' in trajectory[i]:
                    margin_before.append(trajectory[i]['margin'])

            # Get baseline values (early in training, after stabilization)
            baseline_start = min(10, len(trajectory) // 4)
            baseline_end = min(20, len(trajectory) // 2)
            for i in range(baseline_start, baseline_end):
                if i < len(trajectory):
                    if 'loss_variance' in trajectory[i]:
                        loss_var_baseline.append(trajectory[i]['loss_variance'])
                    if 'margin' in trajectory[i]:
                        margin_baseline.append(trajectory[i]['margin'])

    result = {'n_collapses_analyzed': len(collapse_indices)}

    if loss_var_before and loss_var_baseline:
        result['loss_var_before_collapse'] = np.mean(loss_var_before)
        result['loss_var_baseline'] = np.mean(loss_var_baseline)
        result['loss_var_ratio'] = np.mean(loss_var_before) / (np.mean(loss_var_baseline) + 1e-8)

    if margin_before and margin_baseline:
        result['margin_before_collapse'] = np.mean(margin_before)
        result['margin_baseline'] = np.mean(margin_baseline)
        result['margin_ratio'] = np.mean(margin_before) / (np.mean(margin_baseline) + 1e-8)

    return result


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_trajectory_analysis(
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
        heads_to_test = [1, 2, 3]  # Focus on partial damage

    if quick_test:
        baseline_max_steps = 5000
        recovery_steps = 3000
        target_accuracy = 0.95
        seeds = [42, 43]
        print("\n*** QUICK TEST MODE ***\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print("EXPERIMENT 6.3: Time-to-Collapse Trajectory Analysis")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Heads to test: {heads_to_test}")
    print(f"Logging every 100 steps")
    print()

    all_results = {}
    trajectory_classifications = []

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
        model = ModularArithmeticTransformer(p=p, d_model=d_model, n_heads=n_heads,
                                              n_layers=n_layers, d_ff=d_ff)
        baseline_metrics, _ = train_baseline(
            model, train_loader, val_loader, device,
            lr=lr, weight_decay=weight_decay,
            max_steps=baseline_max_steps, target_accuracy=target_accuracy
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

            # Recovery training with dense logging
            trajectory = train_recovery_dense(
                model, train_loader, val_loader, device,
                lr=lr, weight_decay=weight_decay,
                n_steps=recovery_steps,
                log_interval=100,
                desc=f"S{seed}-{n_heads_reinit}H"
            )

            # Classify trajectory
            classification = classify_trajectory(trajectory, baseline_acc)

            # Analyze precursors
            precursor_analysis = analyze_precursors(
                trajectory,
                classification['collapse_indices']
            )

            print(f"  Category: {classification['category']}")
            print(f"  Final: {classification['final_acc']*100:.1f}%")
            print(f"  Collapses: {classification['n_collapses']}")
            if classification['first_collapse_step']:
                print(f"  First collapse at step: {classification['first_collapse_step']}")

            all_results[seed]['conditions'][n_heads_reinit] = {
                'trajectory': trajectory,
                'classification': classification,
                'precursor_analysis': precursor_analysis
            }

            trajectory_classifications.append({
                'seed': seed,
                'n_heads': n_heads_reinit,
                'category': classification['category'],
                'final_acc': classification['final_acc'],
                'n_collapses': classification['n_collapses'],
                'first_collapse_step': classification['first_collapse_step'],
                'min_acc': classification['min_acc'],
                'min_acc_step': classification['min_acc_step']
            })

    # ==========================================================================
    # AGGREGATE ANALYSIS
    # ==========================================================================

    print("\n" + "=" * 70)
    print("AGGREGATE ANALYSIS")
    print("=" * 70)

    # Category counts by condition
    print("\n--- Trajectory Categories by Condition ---")
    print(f"{'Heads':<8} {'STABLE':<10} {'COLLAPSE-RECOVER':<18} {'IRREVERSIBLE':<14} {'PARTIAL':<10}")
    print("-" * 60)

    category_counts = {}
    for n_h in heads_to_test:
        counts = {'STABLE': 0, 'COLLAPSE_THEN_RECOVER': 0, 'IRREVERSIBLE_COLLAPSE': 0, 'PARTIAL_RECOVERY': 0}
        for tc in trajectory_classifications:
            if tc['n_heads'] == n_h:
                counts[tc['category']] += 1
        category_counts[n_h] = counts
        print(f"{n_h:<8} {counts['STABLE']:<10} {counts['COLLAPSE_THEN_RECOVER']:<18} {counts['IRREVERSIBLE_COLLAPSE']:<14} {counts['PARTIAL_RECOVERY']:<10}")

    # Collapse timing distribution
    print("\n--- Collapse Timing Distribution ---")
    collapse_steps = []
    for tc in trajectory_classifications:
        if tc['first_collapse_step'] is not None:
            collapse_steps.append(tc['first_collapse_step'])

    if collapse_steps:
        print(f"N trajectories with collapse: {len(collapse_steps)}/{len(trajectory_classifications)}")
        print(f"First collapse step - Mean: {np.mean(collapse_steps):.0f}, Std: {np.std(collapse_steps):.0f}")
        print(f"                     - Min: {min(collapse_steps)}, Max: {max(collapse_steps)}")

        # Check for clustering
        q25, q75 = np.percentile(collapse_steps, [25, 75])
        print(f"                     - Q25: {q25:.0f}, Q75: {q75:.0f}")

        iqr = q75 - q25
        if iqr < np.mean(collapse_steps) * 0.3:
            print("  -> Collapse times CLUSTERED (IQR < 30% of mean)")
        else:
            print("  -> Collapse times DISTRIBUTED (IQR >= 30% of mean)")
    else:
        print("No collapses detected in any trajectory.")

    # Precursor analysis summary
    print("\n--- Precursor Signal Analysis ---")
    loss_var_ratios = []
    margin_ratios = []

    for seed in all_results:
        for n_h in all_results[seed]['conditions']:
            pa = all_results[seed]['conditions'][n_h]['precursor_analysis']
            if 'loss_var_ratio' in pa:
                loss_var_ratios.append(pa['loss_var_ratio'])
            if 'margin_ratio' in pa:
                margin_ratios.append(pa['margin_ratio'])

    if loss_var_ratios:
        print(f"Loss variance ratio (before/baseline): {np.mean(loss_var_ratios):.2f} +/- {np.std(loss_var_ratios):.2f}")
        if np.mean(loss_var_ratios) > 1.5:
            print("  -> Loss variance ELEVATED before collapse")
        else:
            print("  -> No consistent loss variance precursor")

    if margin_ratios:
        print(f"Margin ratio (before/baseline): {np.mean(margin_ratios):.2f} +/- {np.std(margin_ratios):.2f}")
        if np.mean(margin_ratios) < 0.7:
            print("  -> Margin COMPRESSED before collapse")
        else:
            print("  -> No consistent margin precursor")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare results for JSON (convert numpy types)
    results_for_json = {
        'config': {
            'p': p, 'd_model': d_model, 'n_heads': n_heads, 'n_layers': n_layers,
            'd_ff': d_ff, 'lr': lr, 'weight_decay': weight_decay,
            'recovery_steps': recovery_steps, 'seeds': seeds,
            'heads_to_test': heads_to_test, 'intervention_layer': intervention_layer
        },
        'trajectory_classifications': trajectory_classifications,
        'category_counts': {str(k): v for k, v in category_counts.items()},
        'collapse_timing': {
            'collapse_steps': collapse_steps,
            'mean': float(np.mean(collapse_steps)) if collapse_steps else None,
            'std': float(np.std(collapse_steps)) if collapse_steps else None
        },
        'precursor_summary': {
            'loss_var_ratios': loss_var_ratios,
            'margin_ratios': margin_ratios
        },
        'run_id': run_id
    }

    # Save summary JSON
    json_path = output_dir / f"exp_trajectory_analysis_{run_id}.json"
    with open(json_path, 'w') as f:
        json.dump(results_for_json, f, indent=2, default=str)
    print(f"\nSummary saved to: {json_path}")

    # Save full trajectories (separate file due to size)
    full_path = output_dir / f"exp_trajectory_analysis_{run_id}_full.json"
    full_results = {}
    for seed in all_results:
        full_results[str(seed)] = {
            'baseline_acc': all_results[seed]['baseline_acc'],
            'conditions': {}
        }
        for n_h in all_results[seed]['conditions']:
            cond = all_results[seed]['conditions'][n_h]
            full_results[str(seed)]['conditions'][str(n_h)] = {
                'trajectory': cond['trajectory'],
                'classification': cond['classification'],
                'precursor_analysis': cond['precursor_analysis']
            }

    with open(full_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"Full trajectories saved to: {full_path}")

    # Generate plots
    plot_path = output_dir / f"exp_trajectory_analysis_{run_id}.png"
    plot_trajectory_analysis(all_results, heads_to_test, str(plot_path))

    return results_for_json, all_results


def plot_trajectory_analysis(all_results: Dict, heads_to_test: List[int], save_path: str):
    """Create visualization of trajectory analysis."""
    n_seeds = len(all_results)
    n_conditions = len(heads_to_test)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: All trajectories overlaid by condition
    ax1 = axes[0, 0]
    colors = {'1': 'red', '2': 'blue', '3': 'green'}

    for seed in all_results:
        for n_h in heads_to_test:
            if n_h in all_results[seed]['conditions']:
                traj = all_results[seed]['conditions'][n_h]['trajectory']
                steps = [t['step'] for t in traj]
                accs = [t['accuracy'] * 100 for t in traj]
                ax1.plot(steps, accs, color=colors.get(str(n_h), 'gray'),
                        alpha=0.5, linewidth=1)

    # Add legend entries
    for n_h in heads_to_test:
        ax1.plot([], [], color=colors.get(str(n_h), 'gray'),
                label=f'{n_h} heads', linewidth=2)

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Recovery Trajectories (All Seeds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # Plot 2: Collapse timing histogram
    ax2 = axes[0, 1]
    collapse_steps = []
    for seed in all_results:
        for n_h in all_results[seed]['conditions']:
            cls = all_results[seed]['conditions'][n_h]['classification']
            if cls['first_collapse_step'] is not None:
                collapse_steps.append(cls['first_collapse_step'])

    if collapse_steps:
        ax2.hist(collapse_steps, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(collapse_steps), color='red', linestyle='--',
                   label=f'Mean: {np.mean(collapse_steps):.0f}')
        ax2.set_xlabel('First Collapse Step')
        ax2.set_ylabel('Count')
        ax2.set_title('Collapse Timing Distribution')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No collapses detected', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)

    # Plot 3: Category distribution by condition
    ax3 = axes[1, 0]
    categories = ['STABLE', 'COLLAPSE_THEN_RECOVER', 'IRREVERSIBLE_COLLAPSE', 'PARTIAL_RECOVERY']
    x = np.arange(len(heads_to_test))
    width = 0.2

    for i, cat in enumerate(categories):
        counts = []
        for n_h in heads_to_test:
            count = 0
            for seed in all_results:
                if n_h in all_results[seed]['conditions']:
                    if all_results[seed]['conditions'][n_h]['classification']['category'] == cat:
                        count += 1
            counts.append(count)
        ax3.bar(x + i * width, counts, width, label=cat.replace('_', '\n'))

    ax3.set_xlabel('Heads Reinitialized')
    ax3.set_ylabel('Count')
    ax3.set_title('Trajectory Categories by Condition')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(heads_to_test)
    ax3.legend(fontsize=8)

    # Plot 4: Margin evolution for sample trajectories
    ax4 = axes[1, 1]

    # Pick one seed and show margin for all conditions
    first_seed = list(all_results.keys())[0]
    for n_h in heads_to_test:
        if n_h in all_results[first_seed]['conditions']:
            traj = all_results[first_seed]['conditions'][n_h]['trajectory']
            steps = [t['step'] for t in traj]
            margins = [t['margin'] for t in traj]
            ax4.plot(steps, margins, color=colors.get(str(n_h), 'gray'),
                    label=f'{n_h} heads', linewidth=2)

    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Margin (logit gap)')
    ax4.set_title(f'Margin Evolution (Seed {first_seed})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Trajectory Analysis Experiment")
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

    run_trajectory_analysis(
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
