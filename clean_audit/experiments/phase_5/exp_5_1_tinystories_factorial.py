"""
EXPERIMENT 5.1: 2×2 Factorial on TinyStories (8-Layer)

Tests whether the G×S decomposition and metastability findings from Phase 4
generalize to natural language modeling at 8 layers.

Design mirrors exp_4_2d_factorial.py:
| | B0: No Penalty | B1: Ortho Penalty |
|---|----------------|-------------------|
| A0: No Reference | CE only | CE + ortho-to-EMA-self |
| A1: Reference | CE + anchor (no penalty) | CE + ortho-to-anchor |

Key differences from Phase 4:
- Task: Next-token prediction on TinyStories (vs modular addition)
- Architecture: 8 layers, 256 d_model, 8 heads (~8M params)
- Metric: Token-level accuracy (excluding padding)
- Training: 50K steps with gradient accumulation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import json
import copy
from datetime import datetime
from tqdm import tqdm

from lib.logging_utils import setup_reproducibility
from lib.part_b_utils import freeze_parameters, get_qk_parameters
from lib.deep_transformer import DeepTransformer
from lib.tinystories_dataset import TinyStoriesDataset, get_tinystories_loaders


def validate(model: nn.Module, dataloader: DataLoader, pad_token_id: int, device) -> float:
    """
    Compute token-level accuracy (excluding padding).
    
    Args:
        model: DeepTransformer model
        dataloader: Validation dataloader
        pad_token_id: Token ID to ignore in accuracy computation
        device: torch device
    
    Returns:
        Accuracy as float in [0, 1]
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # [batch, seq, vocab]
            preds = logits.argmax(dim=-1)  # [batch, seq]
            
            # Ignore padding tokens
            mask = (y != pad_token_id)
            correct += ((preds == y) & mask).sum().item()
            total += mask.sum().item()
    model.train()
    return correct / total if total > 0 else 0.0


def compute_stability_metrics(trajectory: List[Tuple[int, float]], 
                               gen_threshold: float = 0.30,  # Lower for LM
                               collapse_threshold: float = 0.15,
                               stability_threshold: float = 0.25) -> Dict:
    """
    Compute stability metrics from accuracy trajectory.
    
    Note: Thresholds are lower than modular arithmetic because
    LM accuracy is harder (random = 0.002% for 50K vocab).
    """
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
    
    dwell_above = sum(1 for _, acc in trajectory if acc >= stability_threshold)
    
    return {
        'time_to_generalization': time_to_gen,
        'collapse_count': collapse_count,
        'time_to_stability': time_to_stability,
        'stability_achieved': time_to_stability != -1,
        'dwell_above_threshold': dwell_above
    }


def save_checkpoint(all_results: Dict, run_id: str, architecture: Dict, 
                    checkpoint_type: str = 'incremental'):
    """
    Save results incrementally to prevent data loss.
    
    Creates timestamped checkpoints so no data is ever overwritten.
    """
    save_dir = Path("clean_audit/data")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Always save with timestamp to never overwrite
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_data = {
        'run_id': run_id,
        'timestamp': timestamp,
        'checkpoint_type': checkpoint_type,
        'architecture': architecture,
        'results': all_results
    }
    
    # Incremental saves go to a separate file
    if checkpoint_type == 'incremental':
        filename = f"exp_5_1_checkpoint_{run_id}_{timestamp}.json"
    else:
        filename = f"exp_5_1_tinystories_factorial_{run_id}.json"
    
    save_path = save_dir / filename
    
    try:
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"  📁 Checkpoint saved: {filename}")
    except Exception as e:
        print(f"  ⚠️ Failed to save checkpoint: {e}")
        # Try emergency save with minimal formatting
        try:
            emergency_path = save_dir / f"EMERGENCY_{run_id}_{timestamp}.json"
            with open(emergency_path, 'w') as f:
                json.dump(save_data, f)
            print(f"  🚨 Emergency save: {emergency_path}")
        except:
            pass
    
    return save_path


def run_condition(
    condition: str,  # 'ce_only', 'ce_ema', 'ce_anchor_nopen', 'ce_anchor_ortho'
    young_g_state: dict,
    anchor_state: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_steps: int,
    lr: float,
    weight_decay: float,
    ortho_lambda: float,
    vocab_size: int,
    pad_token_id: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    max_seq_len: int,
    device,
    grad_accum_steps: int = 8,
    log_interval: int = 500,
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
    model = DeepTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len
    )
    model.load_state_dict(young_g_state)
    model.to(device)
    freeze_parameters(model, freeze_qk=True)
    
    # Anchor model (frozen, for A1 conditions)
    anchor_model = None
    if 'anchor' in condition:
        anchor_model = DeepTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len
        )
        anchor_model.load_state_dict(anchor_state)
        anchor_model.to(device)
        anchor_model.eval()
        for param in anchor_model.parameters():
            param.requires_grad = False
    
    # EMA for ce_ema condition
    ema_resid = None
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    
    trajectory = []
    cosim_history = []
    train_iter = iter(train_loader)
    lambda_warmup = min(5000, training_steps // 4)
    
    optimizer.zero_grad()
    accum_loss = 0.0
    
    for step in range(training_steps):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        
        # Cross-entropy loss (flatten for LM)
        ce_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
            ignore_index=pad_token_id
        )
        
        # Get residual for orthogonality
        resid = model.cache.get('resid_post', model.cache.get('resid_final'))
        if resid is not None:
            resid_norm = F.normalize(resid.mean(dim=1), dim=-1)  # Pool over sequence
        else:
            resid_norm = None
        
        # Compute ortho loss based on condition
        ortho_loss = torch.tensor(0.0, device=device)
        cosim = 0.0
        
        if condition == 'ce_only':
            loss = ce_loss
            
        elif condition == 'ce_ema' and resid_norm is not None:
            resid_mean = resid_norm.mean(dim=0)
            
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
            
        elif condition == 'ce_anchor_nopen' and resid_norm is not None:
            with torch.no_grad():
                _ = anchor_model(x)
                anchor_resid = anchor_model.cache.get('resid_post', anchor_model.cache.get('resid_final'))
                if anchor_resid is not None:
                    anchor_norm = F.normalize(anchor_resid.mean(dim=1), dim=-1)
                    cosim = (anchor_norm * resid_norm).sum(dim=-1).mean().item()
            loss = ce_loss
            
        elif condition == 'ce_anchor_ortho' and resid_norm is not None:
            with torch.no_grad():
                _ = anchor_model(x)
                anchor_resid = anchor_model.cache.get('resid_post', anchor_model.cache.get('resid_final'))
                if anchor_resid is not None:
                    anchor_norm = F.normalize(anchor_resid.mean(dim=1), dim=-1)
            
            if anchor_resid is not None:
                cosim = (anchor_norm * resid_norm).sum(dim=-1).mean().item()
                
                if step >= lambda_warmup:
                    effective_lambda = ortho_lambda * min(1.0, (step - lambda_warmup) / lambda_warmup)
                    ortho_loss = effective_lambda * torch.abs((anchor_norm * resid_norm).sum(dim=-1).mean())
            
            loss = ce_loss + ortho_loss
        else:
            loss = ce_loss
        
        # Gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()
        accum_loss += loss.item()
        
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_loss = 0.0
        
        if step % log_interval == 0 or step == training_steps - 1:
            val_acc = validate(model, val_loader, pad_token_id, device)
            trajectory.append((step, val_acc))
            cosim_history.append((step, cosim))
            
            if step % (log_interval * 5) == 0:
                print(f"    Step {step}: val_acc={val_acc:.4f}, cosim={cosim:.4f}")
    
    final_acc = validate(model, val_loader, pad_token_id, device)
    metrics = compute_stability_metrics(trajectory)
    metrics['final_acc'] = final_acc
    metrics['cosim_history'] = cosim_history
    
    return trajectory, metrics


def run_factorial(
    n_seeds: int = 6,
    vocab_size: int = 50257,  # GPT-2
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 8,
    max_seq_len: int = 127,  # max_length - 1 for LM (input is tokens[:-1])
    max_warmup_steps: int = 10000,
    freeze_threshold: float = 0.15,  # Lower for LM
    training_steps: int = 50000,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    ortho_lambda: float = 0.3,
    train_samples: int = 100000,
    val_samples: int = 10000,
    start_seed: int = 42,
    device: str = 'cuda',
    grad_accum_steps: int = 8,
    quick_test: bool = False
):
    """Run 2×2 factorial experiment on TinyStories."""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if quick_test:
        print("\n*** QUICK TEST MODE ***")
        n_seeds = 1
        training_steps = 2000
        max_warmup_steps = 1000
        train_samples = 10000
        val_samples = 1000
    
    conditions = ['ce_only', 'ce_ema', 'ce_anchor_nopen', 'ce_anchor_ortho']
    condition_labels = {
        'ce_only': 'A0B0: CE Only',
        'ce_ema': 'A0B1: CE + EMA-Self',
        'ce_anchor_nopen': 'A1B0: Anchor (no penalty)',
        'ce_anchor_ortho': 'A1B1: Anchor + Ortho'
    }
    
    print("\n" + "="*80)
    print("PHASE 5: TINYSTORIES 2×2 FACTORIAL (8-Layer)")
    print(f"Architecture: {n_layers}L, d={d_model}, h={n_heads} (~8M params)")
    print(f"Conditions: {conditions}")
    print(f"Seeds: {n_seeds}, Training: {training_steps} steps")
    print("="*80 + "\n")
    
    # Load data (shared across seeds for consistency)
    print("Loading TinyStories dataset...")
    train_loader, val_loader, actual_vocab_size, pad_token_id = get_tinystories_loaders(
        batch_size=batch_size,
        max_length=max_seq_len + 1,  # +1 because we split into input[:-1] and target[1:]
        train_samples=train_samples,
        val_samples=val_samples,
        seed=42  # Fixed for dataset consistency
    )
    vocab_size = actual_vocab_size
    print(f"  Vocab size: {vocab_size}, Pad token: {pad_token_id}")
    
    # Generate unique run ID for this experiment
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"  Run ID: {run_id}")
    
    # Architecture info for checkpoints
    architecture = {
        'n_layers': n_layers,
        'd_model': d_model,
        'n_heads': n_heads,
        'vocab_size': vocab_size,
        'max_seq_len': max_seq_len,
        'training_steps': training_steps,
        'n_seeds': n_seeds
    }
    
    all_results = {c: [] for c in conditions}
    
    for seed_idx in range(n_seeds):
        seed = start_seed + seed_idx
        print(f"\n>>> SEED {seed} ({seed_idx+1}/{n_seeds})")
        
        setup_reproducibility(seed)
        
        # Warmup to capture Young G
        print("  Capturing Young G...")
        base_model = DeepTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len
        )
        base_model.to(device)
        optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)
        train_iter = iter(train_loader)
        
        consecutive_above = 0
        actual_freeze_step = max_warmup_steps
        
        optimizer.zero_grad()
        for step in tqdm(range(max_warmup_steps), desc="  Warmup"):
            base_model.train()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            logits = base_model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=pad_token_id
            )
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if step % 200 == 0:
                val_acc = validate(base_model, val_loader, pad_token_id, device)
                if val_acc >= freeze_threshold:
                    consecutive_above += 1
                    if consecutive_above >= 2:
                        actual_freeze_step = step
                        print(f"\n  Critical period at step {step} (val_acc={val_acc:.4f})")
                        break
                else:
                    consecutive_above = 0
        
        young_g_state = copy.deepcopy(base_model.state_dict())
        anchor_state = copy.deepcopy(young_g_state)
        
        # Run each condition
        for cond in conditions:
            print(f"\n  Running {condition_labels[cond]}...")
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
                vocab_size=vocab_size,
                pad_token_id=pad_token_id,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                max_seq_len=max_seq_len,
                device=device,
                grad_accum_steps=grad_accum_steps
            )
            
            all_results[cond].append({
                'seed': seed,
                'freeze_step': actual_freeze_step,
                'trajectory': trajectory,
                'metrics': metrics
            })
            
            status = "✓ Stable" if metrics['stability_achieved'] else "✗ Unstable"
            print(f"    {status} | Collapses: {metrics['collapse_count']} | "
                  f"Final: {metrics['final_acc']:.4f} | Dwell: {metrics['dwell_above_threshold']}")
            
            # Incremental save after each condition (crash protection)
            save_checkpoint(all_results, run_id, architecture, checkpoint_type='incremental')
    
    # Aggregate analysis
    print("\n" + "="*80)
    print("PHASE 5 FACTORIAL RESULTS")
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
            'mean_collapses': float(np.mean(collapse_counts)),
            'std_collapses': float(np.std(collapse_counts)),
            'mean_acc': float(np.mean(final_accs)),
            'std_acc': float(np.std(final_accs)),
            'mean_dwell': float(np.mean(dwells)),
            'std_dwell': float(np.std(dwells))
        }
        
        print(f"{condition_labels[cond]:<25} {stability_rate*100:>4.0f}%{'':<6} "
              f"{np.mean(collapse_counts):>5.1f} ± {np.std(collapse_counts):<5.1f} "
              f"{np.mean(final_accs):>5.4f} ± {np.std(final_accs):<5.4f} "
              f"{np.mean(dwells):>5.1f}")
    
    # 2×2 analysis
    print("\n" + "-"*40)
    print("MAIN EFFECTS (2×2 ANOVA-style)")
    print("-"*40)
    
    a0_stable = (summary['ce_only']['stability_rate'] + summary['ce_ema']['stability_rate']) / 2
    a1_stable = (summary['ce_anchor_nopen']['stability_rate'] + summary['ce_anchor_ortho']['stability_rate']) / 2
    print(f"Reference effect: A0={a0_stable*100:.0f}% vs A1={a1_stable*100:.0f}% (Δ={100*(a1_stable-a0_stable):+.0f}%)")
    
    b0_stable = (summary['ce_only']['stability_rate'] + summary['ce_anchor_nopen']['stability_rate']) / 2
    b1_stable = (summary['ce_ema']['stability_rate'] + summary['ce_anchor_ortho']['stability_rate']) / 2
    print(f"Penalty effect:   B0={b0_stable*100:.0f}% vs B1={b1_stable*100:.0f}% (Δ={100*(b1_stable-b0_stable):+.0f}%)")
    
    # Final save with unique run_id
    save_data = {
        'run_id': run_id,
        'completed_at': datetime.now().isoformat(),
        'conditions': conditions,
        'n_seeds': n_seeds,
        'architecture': architecture,
        'results': all_results,
        'summary': summary
    }
    
    # Save both with run_id (primary) and as latest (for convenience)
    save_path = Path(f"clean_audit/data/exp_5_1_tinystories_factorial_{run_id}.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n✅ Final results saved to: {save_path}")
    
    # Also save as "latest" for convenience
    latest_path = Path("clean_audit/data/exp_5_1_tinystories_factorial_LATEST.json")
    with open(latest_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"📌 Also saved as: {latest_path}")
    
    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description="Phase 5: TinyStories 2×2 Factorial (8-Layer)")
    parser.add_argument('--n_seeds', type=int, default=6)
    parser.add_argument('--training_steps', type=int, default=50000)
    parser.add_argument('--ortho_lambda', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick_test', action='store_true', help='Run quick sanity test')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=8)
    
    args = parser.parse_args()
    
    run_factorial(
        n_seeds=args.n_seeds,
        training_steps=args.training_steps,
        ortho_lambda=args.ortho_lambda,
        start_seed=args.seed,
        device=args.device,
        quick_test=args.quick_test,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers
    )


if __name__ == '__main__':
    main()
