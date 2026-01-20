"""
Experiment: Recoverability After Geometry Destruction in Modular Arithmetic Transformers

Motivation:
    We observe two regimes after geometry perturbations:
    - Some geometry breaks degrade performance but remain trainable
    - Others cause catastrophic collapse to chance

    This experiment tests whether catastrophic geometry breaks destroy task-aligned
    gradient structure, resulting in recovery that is dramatically slower compared
    to milder, recoverable breaks.

Core Hypothesis:
    There exist qualitatively distinct regimes of geometry mismatch:
    - Recoverable regime: task-aligned gradient structure is preserved, learning resumes quickly
    - Collapsed regime: task-aligned gradient structure is erased, orders-of-magnitude slower recovery

Protocol:
    1. Train 3-layer transformer on modular addition (p=113) to ≥99% val accuracy
    2. Save checkpoint A (fully converged, consolidated geometry)
    3. Create two intervention conditions:
       - Condition 1 (Recoverable): Reinit Q/K of ONE head in layer 1
       - Condition 2 (Catastrophic): Reinit ALL Q/K in layer 1
    4. Resume training for 20k steps with identical optimizer settings
    5. Compare recovery dynamics

Metrics:
    - Validation accuracy vs training step
    - Training loss vs training step
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
# MODEL ARCHITECTURE (3-Layer Transformer for Modular Addition)
# =============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block with pre-LN architecture."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention
        h_ln = self.ln1(x)
        attn_out, _ = self.attn(h_ln, h_ln, h_ln)
        x = x + attn_out

        # Pre-LN FFN
        h_ln = self.ln2(x)
        x = x + self.ffn(h_ln)

        return x


class ModularArithmeticTransformer(nn.Module):
    """
    3-Layer Transformer for modular addition task.

    Architecture (from experiment brief):
        - Layers: 3
        - d_model: 128
        - Attention heads: 4
        - MLP hidden size: 512
        - Pre-LayerNorm
        - Learned token embeddings + positional encodings

    Input format: [a, "+", b] -> (a + b) mod p
    Output: single classification token (0 to p-1)
    """

    def __init__(
        self,
        p: int = 113,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()

        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Vocabulary: a tokens (0 to p-1) + operator token + b tokens (p to 2p-1)
        # Total vocab = 2*p + 1 (the +1 for the "+" operator)
        self.vocab_size = 2 * p + 1
        self.plus_token = p  # "+" is token p

        # Embeddings
        self.embed = nn.Embedding(self.vocab_size, d_model)
        self.pos_embed = nn.Embedding(3, d_model)  # 3 positions: a, +, b

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)  # Classify into 0..p-1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 3] with tokens [a, "+", b_shifted]
               where a in [0, p-1], "+" = p, b_shifted in [p+1, 2p]

        Returns:
            logits: [batch, p] classification over [0, p-1]
        """
        B, L = x.shape

        # Embeddings
        tok_emb = self.embed(x)
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)
        h = tok_emb + pos_emb

        # Process through blocks
        for block in self.blocks:
            h = block(h)

        # Output from last position
        h_final = self.ln_final(h[:, -1, :])
        logits = self.head(h_final)

        return logits


# =============================================================================
# DATASET
# =============================================================================

class ModularAdditionDataset(Dataset):
    """
    Modular addition task: (a + b) mod p

    Input: [a, "+", b] as token sequence
    Target: (a + b) mod p

    Uses fixed train/val split (50/50) for grokking setup.
    """

    def __init__(
        self,
        p: int = 113,
        train_fraction: float = 0.5,
        is_train: bool = True,
        seed: int = 42
    ):
        self.p = p
        self.plus_token = p  # "+" operator token

        # Generate all (a, b) pairs
        all_pairs = [(a, b) for a in range(p) for b in range(p)]

        # Shuffle and split
        rng = np.random.RandomState(seed)
        rng.shuffle(all_pairs)

        n_train = int(len(all_pairs) * train_fraction)

        if is_train:
            self.pairs = all_pairs[:n_train]
        else:
            self.pairs = all_pairs[n_train:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        target = (a + b) % self.p

        # Input sequence: [a, "+", b_shifted]
        # a is in [0, p-1]
        # "+" is token p
        # b_shifted is b + p + 1 (to separate from a tokens)
        input_seq = torch.LongTensor([a, self.plus_token, b + self.p + 1])

        return input_seq, torch.tensor(target, dtype=torch.long)


# =============================================================================
# QK REINITIALIZATION INTERVENTIONS
# =============================================================================

def reinit_qk_single_head(model: nn.Module, layer_idx: int, head_idx: int):
    """
    Reinitialize Q and K matrices for a SINGLE attention head.

    Condition 1: Recoverable geometry break
    - Localized routing disruption
    - Expected: significant drop but above chance

    Args:
        model: The transformer model
        layer_idx: Which layer (0-indexed)
        head_idx: Which head within that layer (0-indexed)
    """
    block = model.blocks[layer_idx]
    attn = block.attn

    d_model = attn.embed_dim
    n_heads = attn.num_heads
    head_dim = d_model // n_heads

    # in_proj_weight has shape [3*d_model, d_model]
    # Layout: [Q; K; V] stacked vertically
    # Each of Q, K, V has shape [d_model, d_model]
    # For head i, the rows are [i*head_dim : (i+1)*head_dim]

    start_row = head_idx * head_dim
    end_row = (head_idx + 1) * head_dim

    with torch.no_grad():
        # Reinit Q for this head (rows start_row:end_row of first d_model rows)
        nn.init.xavier_uniform_(
            attn.in_proj_weight[start_row:end_row, :].unsqueeze(0)
        )
        attn.in_proj_weight[start_row:end_row, :] = torch.nn.init.xavier_uniform_(
            torch.empty(head_dim, d_model, device=attn.in_proj_weight.device)
        )

        # Reinit K for this head (rows d_model + start_row : d_model + end_row)
        k_start = d_model + start_row
        k_end = d_model + end_row
        attn.in_proj_weight[k_start:k_end, :] = torch.nn.init.xavier_uniform_(
            torch.empty(head_dim, d_model, device=attn.in_proj_weight.device)
        )

        # Reinit biases if they exist
        if attn.in_proj_bias is not None:
            nn.init.zeros_(attn.in_proj_bias[start_row:end_row])
            nn.init.zeros_(attn.in_proj_bias[k_start:k_end])

    print(f"  Reinitialized Q/K for layer {layer_idx}, head {head_idx}")
    print(f"    Q rows: [{start_row}:{end_row}], K rows: [{k_start}:{k_end}]")


def reinit_qk_all_heads(model: nn.Module, layer_idx: int):
    """
    Reinitialize ALL Q and K matrices in a layer.

    Condition 2: Catastrophic geometry break
    - Global routing disruption for entire layer
    - Expected: collapse to near chance (~1/113)

    Args:
        model: The transformer model
        layer_idx: Which layer (0-indexed)
    """
    block = model.blocks[layer_idx]
    attn = block.attn

    d_model = attn.embed_dim

    with torch.no_grad():
        # Reinit all of Q (rows 0:d_model)
        attn.in_proj_weight[:d_model, :] = torch.nn.init.xavier_uniform_(
            torch.empty(d_model, d_model, device=attn.in_proj_weight.device)
        )

        # Reinit all of K (rows d_model:2*d_model)
        attn.in_proj_weight[d_model:2*d_model, :] = torch.nn.init.xavier_uniform_(
            torch.empty(d_model, d_model, device=attn.in_proj_weight.device)
        )

        # Reinit biases if they exist
        if attn.in_proj_bias is not None:
            nn.init.zeros_(attn.in_proj_bias[:d_model])  # Q bias
            nn.init.zeros_(attn.in_proj_bias[d_model:2*d_model])  # K bias

    print(f"  Reinitialized ALL Q/K for layer {layer_idx}")
    print(f"    Q rows: [0:{d_model}], K rows: [{d_model}:{2*d_model}]")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def validate(model: nn.Module, dataloader: DataLoader, device: str) -> Dict[str, float]:
    """Compute validation accuracy and loss."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_loss += loss.item() * inputs.shape[0]
            total_samples += inputs.shape[0]

    return {
        'accuracy': total_correct / total_samples,
        'loss': total_loss / total_samples
    }


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    max_steps: int = 50000,
    target_accuracy: float = 0.99,
    log_interval: int = 500
) -> Tuple[Dict, List[Dict]]:
    """
    Train model to convergence (≥99% val accuracy).

    Returns:
        (final_metrics, trajectory)
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_iter = iter(train_loader)
    trajectory = []

    print(f"\nBaseline Training (target: {target_accuracy*100:.0f}% val acc)")
    print("-" * 60)

    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:
        model.train()

        # Get batch
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward/backward
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        # Logging
        if (step + 1) % log_interval == 0 or step == 0:
            metrics = validate(model, val_loader, device)
            metrics['step'] = step + 1
            metrics['train_loss'] = loss.item()
            trajectory.append(metrics)

            pbar.set_postfix({
                'val_acc': f"{metrics['accuracy']*100:.2f}%",
                'loss': f"{metrics['loss']:.4f}"
            })

            # Early stopping if converged
            if metrics['accuracy'] >= target_accuracy:
                print(f"\n[OK] Converged at step {step+1} with {metrics['accuracy']*100:.2f}% val accuracy")
                return metrics, trajectory

    final_metrics = validate(model, val_loader, device)
    final_metrics['step'] = max_steps
    print(f"\nReached max steps. Final val accuracy: {final_metrics['accuracy']*100:.2f}%")
    return final_metrics, trajectory


def train_recovery(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    n_steps: int = 20000,
    log_interval: int = 100,
    condition_name: str = "recovery"
) -> List[Dict]:
    """
    Recovery training phase after intervention.

    Uses the SAME optimizer (do not reset optimizer state).
    Logs metrics at regular intervals.

    Returns:
        trajectory: List of {step, accuracy, loss, train_loss} dicts
    """
    train_iter = iter(train_loader)
    trajectory = []

    print(f"\nRecovery Training: {condition_name}")
    print("-" * 60)

    # Log initial state (step 0)
    metrics = validate(model, val_loader, device)
    metrics['step'] = 0
    metrics['train_loss'] = float('nan')
    trajectory.append(metrics)
    print(f"  Initial: val_acc={metrics['accuracy']*100:.2f}%, loss={metrics['loss']:.4f}")

    pbar = tqdm(range(n_steps), desc=condition_name)
    for step in pbar:
        model.train()

        # Get batch
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward/backward
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        # Logging
        if (step + 1) % log_interval == 0:
            metrics = validate(model, val_loader, device)
            metrics['step'] = step + 1
            metrics['train_loss'] = loss.item()
            trajectory.append(metrics)

            pbar.set_postfix({
                'val_acc': f"{metrics['accuracy']*100:.2f}%",
                'loss': f"{metrics['loss']:.4f}"
            })

    return trajectory


# =============================================================================
# PLOTTING
# =============================================================================

def plot_recovery_comparison(
    baseline_trajectory: List[Dict],
    cond1_trajectory: List[Dict],
    cond2_trajectory: List[Dict],
    save_path: str,
    p: int = 113
):
    """
    Plot recovery dynamics comparison.

    Two subplots:
    1. Validation accuracy vs step
    2. Training loss vs step
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    chance_level = 1.0 / p

    # Extract data
    def extract(traj, key):
        return [m['step'] for m in traj], [m[key] for m in traj]

    # Plot 1: Validation Accuracy
    ax1 = axes[0]

    steps, accs = extract(cond1_trajectory, 'accuracy')
    ax1.plot(steps, [a * 100 for a in accs], 'b-', linewidth=2,
             label='Cond 1: Single Head (Recoverable)', marker='o', markersize=3)

    steps, accs = extract(cond2_trajectory, 'accuracy')
    ax1.plot(steps, [a * 100 for a in accs], 'r-', linewidth=2,
             label='Cond 2: All Heads (Catastrophic)', marker='s', markersize=3)

    ax1.axhline(y=chance_level * 100, color='gray', linestyle='--',
                label=f'Chance ({chance_level*100:.2f}%)')
    ax1.axhline(y=99.0, color='green', linestyle=':', alpha=0.7,
                label='Converged (99%)')

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax1.set_title('Recovery Dynamics: Validation Accuracy', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # Plot 2: Training Loss
    ax2 = axes[1]

    steps, losses = extract(cond1_trajectory, 'loss')
    ax2.plot(steps, losses, 'b-', linewidth=2,
             label='Cond 1: Single Head', marker='o', markersize=3)

    steps, losses = extract(cond2_trajectory, 'loss')
    ax2.plot(steps, losses, 'r-', linewidth=2,
             label='Cond 2: All Heads', marker='s', markersize=3)

    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Recovery Dynamics: Validation Loss', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to: {save_path}")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(
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
    intervention_head: int = 0,
    quick_test: bool = False
):
    """
    Run the full recovery dynamics experiment.
    """
    if quick_test:
        baseline_max_steps = 5000
        recovery_steps = 2000
        target_accuracy = 0.90
        print("\n*** QUICK TEST MODE ***\n")

    setup_reproducibility(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print("EXPERIMENT: Recoverability After Geometry Destruction")
    print("=" * 70)
    print(f"\nRun ID: {run_id}")
    print(f"Device: {device}")
    print(f"\nModel Configuration:")
    print(f"  Layers: {n_layers}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"\nTask: Modular Addition (p={p})")
    print(f"Chance level: {100/p:.2f}%")
    print(f"\nIntervention Layer: {intervention_layer}")
    print(f"Intervention Head (Cond 1): {intervention_head}")

    # Create datasets
    train_dataset = ModularAdditionDataset(p=p, is_train=True, seed=seed)
    val_dataset = ModularAdditionDataset(p=p, is_train=False, seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nDataset:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # =========================================================================
    # PHASE 1: Baseline Training to Convergence
    # =========================================================================

    print("\n" + "=" * 70)
    print("PHASE 1: Baseline Training")
    print("=" * 70)

    model = ModularArithmeticTransformer(
        p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    baseline_metrics, baseline_trajectory = train_baseline(
        model, train_loader, val_loader, device,
        lr=lr, weight_decay=weight_decay,
        max_steps=baseline_max_steps,
        target_accuracy=target_accuracy
    )

    if baseline_metrics['accuracy'] < target_accuracy:
        print(f"\n[!] WARNING: Baseline did not reach target accuracy!")
        print(f"  Achieved: {baseline_metrics['accuracy']*100:.2f}%")
        print(f"  Target: {target_accuracy*100:.0f}%")

    # Save checkpoint A (converged model + optimizer state)
    checkpoint_a = {
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'baseline_metrics': baseline_metrics,
        'baseline_trajectory': baseline_trajectory
    }

    # =========================================================================
    # PHASE 2: Condition 1 - Recoverable Geometry Break (Single Head)
    # =========================================================================

    print("\n" + "=" * 70)
    print("PHASE 2: Condition 1 - Recoverable Geometry Break")
    print("=" * 70)

    # Restore from checkpoint A
    model.load_state_dict(copy.deepcopy(checkpoint_a['model_state_dict']))
    model.to(device)

    # Create fresh optimizer (as per protocol: "Do not reset optimizer state"
    # refers to not resetting DURING recovery, but we need a new optimizer
    # for each condition starting fresh from the checkpoint)
    optimizer_cond1 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Apply intervention
    print("\nApplying intervention:")
    reinit_qk_single_head(model, layer_idx=intervention_layer, head_idx=intervention_head)

    # Measure immediate impact
    post_intervention = validate(model, val_loader, device)
    print(f"\nPost-intervention accuracy: {post_intervention['accuracy']*100:.2f}%")
    print(f"Drop from baseline: {(baseline_metrics['accuracy'] - post_intervention['accuracy'])*100:.2f}%")

    # Recovery training
    cond1_trajectory = train_recovery(
        model, optimizer_cond1, train_loader, val_loader, device,
        n_steps=recovery_steps,
        log_interval=100 if not quick_test else 50,
        condition_name="Cond 1: Single Head"
    )

    cond1_results = {
        'name': 'Recoverable (Single Head)',
        'intervention': f'Layer {intervention_layer}, Head {intervention_head}',
        'post_intervention_acc': post_intervention['accuracy'],
        'trajectory': cond1_trajectory,
        'final_acc': cond1_trajectory[-1]['accuracy']
    }

    # =========================================================================
    # PHASE 3: Condition 2 - Catastrophic Geometry Break (All Heads)
    # =========================================================================

    print("\n" + "=" * 70)
    print("PHASE 3: Condition 2 - Catastrophic Geometry Break")
    print("=" * 70)

    # Restore from checkpoint A
    model.load_state_dict(copy.deepcopy(checkpoint_a['model_state_dict']))
    model.to(device)

    # Create fresh optimizer
    optimizer_cond2 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Apply intervention
    print("\nApplying intervention:")
    reinit_qk_all_heads(model, layer_idx=intervention_layer)

    # Measure immediate impact
    post_intervention = validate(model, val_loader, device)
    print(f"\nPost-intervention accuracy: {post_intervention['accuracy']*100:.2f}%")
    print(f"Drop from baseline: {(baseline_metrics['accuracy'] - post_intervention['accuracy'])*100:.2f}%")

    # Recovery training
    cond2_trajectory = train_recovery(
        model, optimizer_cond2, train_loader, val_loader, device,
        n_steps=recovery_steps,
        log_interval=100 if not quick_test else 50,
        condition_name="Cond 2: All Heads"
    )

    cond2_results = {
        'name': 'Catastrophic (All Heads)',
        'intervention': f'Layer {intervention_layer}, All Heads',
        'post_intervention_acc': post_intervention['accuracy'],
        'trajectory': cond2_trajectory,
        'final_acc': cond2_trajectory[-1]['accuracy']
    }

    # =========================================================================
    # ANALYSIS AND OUTPUT
    # =========================================================================

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print(f"\nBaseline converged accuracy: {baseline_metrics['accuracy']*100:.2f}%")
    print(f"\nCondition 1 (Recoverable - Single Head):")
    print(f"  Post-intervention: {cond1_results['post_intervention_acc']*100:.2f}%")
    print(f"  Final (after {recovery_steps} steps): {cond1_results['final_acc']*100:.2f}%")

    print(f"\nCondition 2 (Catastrophic - All Heads):")
    print(f"  Post-intervention: {cond2_results['post_intervention_acc']*100:.2f}%")
    print(f"  Final (after {recovery_steps} steps): {cond2_results['final_acc']*100:.2f}%")

    # Recovery rate analysis
    def compute_recovery_metrics(trajectory, baseline_acc):
        """Compute recovery-related metrics."""
        # Time to reach 50% of baseline
        half_target = baseline_acc * 0.5
        steps_to_half = None
        for m in trajectory:
            if m['accuracy'] >= half_target:
                steps_to_half = m['step']
                break

        # Time to reach 90% of baseline
        ninety_target = baseline_acc * 0.9
        steps_to_ninety = None
        for m in trajectory:
            if m['accuracy'] >= ninety_target:
                steps_to_ninety = m['step']
                break

        # Final recovery percentage
        final_recovery = trajectory[-1]['accuracy'] / baseline_acc if baseline_acc > 0 else 0

        return {
            'steps_to_50pct': steps_to_half,
            'steps_to_90pct': steps_to_ninety,
            'final_recovery_pct': final_recovery
        }

    cond1_recovery = compute_recovery_metrics(cond1_trajectory, baseline_metrics['accuracy'])
    cond2_recovery = compute_recovery_metrics(cond2_trajectory, baseline_metrics['accuracy'])

    print(f"\nRecovery Metrics:")
    print(f"  Cond 1 - Steps to 50% baseline: {cond1_recovery['steps_to_50pct']}")
    print(f"  Cond 1 - Steps to 90% baseline: {cond1_recovery['steps_to_90pct']}")
    print(f"  Cond 1 - Final recovery: {cond1_recovery['final_recovery_pct']*100:.1f}%")
    print(f"  Cond 2 - Steps to 50% baseline: {cond2_recovery['steps_to_50pct']}")
    print(f"  Cond 2 - Steps to 90% baseline: {cond2_recovery['steps_to_90pct']}")
    print(f"  Cond 2 - Final recovery: {cond2_recovery['final_recovery_pct']*100:.1f}%")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    # Detect unexpected outcomes
    cond1_collapsed = cond1_results['final_acc'] < 0.5 * baseline_metrics['accuracy']
    cond2_recovered = cond2_results['final_acc'] > 0.9 * baseline_metrics['accuracy']

    if cond1_collapsed and cond2_recovered:
        print("\n[!!] UNEXPECTED RESULT: Opposite of hypothesis!")
        print("  - Condition 1 (single head) COLLAPSED during recovery")
        print("  - Condition 2 (all heads) RECOVERED successfully")
        print("\nPossible interpretation:")
        print("  The partial QK reinit created a G/S mismatch that destabilized")
        print("  during training, while the full reinit allowed clean relearning.")
        print("  This suggests the problem isn't gradient structure erasure,")
        print("  but rather interference between old and new geometry.")
    elif cond1_recovery['steps_to_90pct'] is not None and cond2_recovery['steps_to_90pct'] is None:
        print("\n[OK] Results support hypothesis:")
        print("  - Condition 1 recovered to 90% baseline")
        print("  - Condition 2 did NOT recover to 90% baseline")
        print("  -> Catastrophic geometry breaks appear to erase task-aligned gradient structure")
    elif cond1_recovery['steps_to_90pct'] is not None and cond2_recovery['steps_to_90pct'] is not None:
        # Handle division by zero when cond1 starts above 90%
        if cond1_recovery['steps_to_90pct'] == 0:
            print(f"\nCondition 1 started above 90% baseline (minimal damage)")
            print(f"Condition 2 recovered in {cond2_recovery['steps_to_90pct']} steps")
        else:
            ratio = cond2_recovery['steps_to_90pct'] / cond1_recovery['steps_to_90pct']
            print(f"\nBoth conditions recovered. Speed ratio: {ratio:.1f}x")
            if ratio > 5:
                print("[OK] Condition 2 was >5x slower - supports distinct regimes hypothesis")
            else:
                print("[!] Recovery speeds similar - may not support distinct regimes")
    else:
        print("\n[!] Neither condition reached 90% - experiment may need longer recovery phase")

    # Save results
    results = {
        'config': {
            'p': p,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'lr': lr,
            'weight_decay': weight_decay,
            'recovery_steps': recovery_steps,
            'seed': seed,
            'intervention_layer': intervention_layer
        },
        'baseline': {
            'accuracy': baseline_metrics['accuracy'],
            'trajectory': baseline_trajectory
        },
        'condition_1': cond1_results,
        'condition_2': cond2_results,
        'recovery_metrics': {
            'condition_1': cond1_recovery,
            'condition_2': cond2_recovery
        },
        'run_id': run_id
    }

    # Save JSON
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"exp_recovery_dynamics_{run_id}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Save plot
    plot_path = output_dir / f"exp_recovery_dynamics_{run_id}.png"
    plot_recovery_comparison(
        baseline_trajectory,
        cond1_trajectory,
        cond2_trajectory,
        str(plot_path),
        p=p
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Recoverability After Geometry Destruction Experiment"
    )

    # Model config
    parser.add_argument('--p', type=int, default=113, help='Prime modulus')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--d_ff', type=int, default=512, help='FFN hidden dimension')

    # Training config
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1.0, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--baseline_steps', type=int, default=50000, help='Max baseline training steps')
    parser.add_argument('--recovery_steps', type=int, default=20000, help='Recovery training steps')
    parser.add_argument('--target_accuracy', type=float, default=0.99, help='Target baseline accuracy')

    # Experiment config
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--intervention_layer', type=int, default=1, help='Layer for intervention (0-indexed)')
    parser.add_argument('--intervention_head', type=int, default=0, help='Head for Cond 1 (0-indexed)')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with reduced steps')

    args = parser.parse_args()

    run_experiment(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        baseline_max_steps=args.baseline_steps,
        recovery_steps=args.recovery_steps,
        target_accuracy=args.target_accuracy,
        seed=args.seed,
        intervention_layer=args.intervention_layer,
        intervention_head=args.intervention_head,
        quick_test=args.quick_test
    )


if __name__ == '__main__':
    main()
