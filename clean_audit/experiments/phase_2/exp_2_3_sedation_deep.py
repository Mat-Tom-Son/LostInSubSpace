"""
EXPERIMENT 2.3: Sedation Test (2-Layer)

Phase 2: Prove margin-as-budget story holds at 2 layers.

Protocol:
1. Load converged 2-layer model (from Exp 2.1 or train fresh)
2. Clamp output amplitude to 60% of natural
3. Test clean (σ=0) vs noisy (σ=2.0) accuracy

Success Criteria:
- Clean accuracy: minimal degradation (<5%)
- Noisy accuracy: drops >20% vs natural amplitude
- Confirms margin is prophylactic buffer, not computational necessity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional
import argparse
import json

from lib.logging_utils import setup_reproducibility
from lib.clamps import NaiveClamp
from lib.deep_transformer import DeepTransformer
from experiments.exp_a_foundation import InterleavedSequenceDataset


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_steps: int,
    lr: float,
    device: str,
    vocab_size: int
) -> Dict:
    """Train model to convergence for sedation testing."""
    
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_iter = iter(train_loader)
    history = []
    
    for step in range(n_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        
        if step % 1000 == 0 or step == n_steps - 1:
            val_acc = validate(model, val_loader, device, vocab_size)
            history.append({'step': step, 'loss': loss.item(), 'val_acc': val_acc})
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.3f}")
    
    return {'history': history, 'final_acc': history[-1]['val_acc']}


def validate(model: nn.Module, dataloader: DataLoader, device: str, vocab_size: int) -> float:
    """Validation accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    
    model.train()
    return correct / total if total > 0 else 0.0


def measure_natural_amplitude(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> float:
    """Measure natural amplitude (post-LN residual norm)."""
    model.eval()
    amplitudes = []
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            
            if 'resid_final' in model.cache:
                resid = model.cache['resid_final']
                amp = resid.reshape(-1, resid.shape[-1]).norm(dim=-1).mean().item()
                amplitudes.append(amp)
    
    return np.mean(amplitudes)


def evaluate_with_sedation(
    model: nn.Module,
    dataloader: DataLoader,
    clamp_target: Optional[float],
    noise_scale: float,
    device: str,
    vocab_size: int
) -> Dict[str, float]:
    """
    Evaluate model with optional clamp and post-clamp noise.
    
    Args:
        model: Trained model
        dataloader: Validation data
        clamp_target: Target norm for clamp (None = no clamp)
        noise_scale: Post-clamp noise σ
        device: Compute device
        vocab_size: Vocabulary size
    """
    model.eval()
    
    # Set up clamp
    if clamp_target is not None:
        clamp = NaiveClamp(target_norm=clamp_target, layer_idx=0)
        model.clamp_fn = clamp
    else:
        model.clamp_fn = None
    
    # Set post-clamp noise
    model.post_clamp_noise_scale = noise_scale
    
    correct = 0
    total = 0
    margins = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits = model(inputs)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
            
            # Compute margins (for analysis)
            # Flatten for margin computation
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            
            true_logits = logits_flat.gather(1, targets_flat.unsqueeze(1)).squeeze()
            max_other = logits_flat.scatter(1, targets_flat.unsqueeze(1), float('-inf')).max(dim=1).values
            margin = (true_logits - max_other).mean().item()
            margins.append(margin)
    
    # Clean up
    model.clamp_fn = None
    model.post_clamp_noise_scale = 0.0
    
    return {
        'accuracy': correct / total if total > 0 else 0.0,
        'mean_margin': np.mean(margins) if margins else 0.0,
        'n_samples': total
    }


def run_single_seed(
    seq_len: int = 128,
    vocab_size: int = 4096,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    n_steps: int = 15000,
    batch_size: int = 64,
    lr: float = 1e-3,
    clamp_fraction: float = 0.6,
    noise_level: float = 2.0,
    seed: int = 42,
    device: str = 'cuda'
) -> Dict:
    """
    Run Experiment 2.3: Sedation Test on 2-Layer Model.
    
    Protocol:
    1. Train 2-layer model to convergence
    2. Measure natural amplitude
    3. Clamp to clamp_fraction of natural
    4. Compare clean vs noisy accuracy
    """
    
    print("\n" + "="*80)
    print(f"EXPERIMENT 2.3: SEDATION TEST (2-LAYER, seed={seed})")
    print("="*80 + "\n")
    
    setup_reproducibility(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Architecture: n_layers={n_layers}, d_model={d_model}")
    print(f"Clamp fraction: {clamp_fraction:.0%}, Noise: σ={noise_level}\n")
    
    # Data
    train_dataset = InterleavedSequenceDataset(
        n_samples=n_steps * batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size
    )
    val_dataset = InterleavedSequenceDataset(
        n_samples=5000,
        seq_len=seq_len,
        vocab_size=vocab_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # =========================================================================
    # PHASE 1: Train Model
    # =========================================================================
    
    print("-"*80)
    print("PHASE 1: Training 2-Layer Model")
    print("-"*80 + "\n")
    
    model = DeepTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=seq_len,
        dropout=0.0
    )
    
    train_result = train_model(
        model, train_loader, val_loader,
        n_steps=n_steps, lr=lr, device=device, vocab_size=vocab_size
    )
    
    print(f"\nTraining complete. Final accuracy: {train_result['final_acc']:.3f}")
    
    # =========================================================================
    # PHASE 2: Measure Natural Amplitude
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 2: Measuring Natural Amplitude")
    print("-"*80 + "\n")
    
    natural_amplitude = measure_natural_amplitude(model, val_loader, device)
    clamp_target = natural_amplitude * clamp_fraction
    
    print(f"Natural amplitude: {natural_amplitude:.2f}")
    print(f"Clamp target ({clamp_fraction:.0%}): {clamp_target:.2f}")
    
    # =========================================================================
    # PHASE 3: Sedation Tests
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 3: Sedation Tests")
    print("-"*80 + "\n")
    
    results = {
        'config': {
            'n_layers': n_layers,
            'd_model': d_model,
            'n_heads': n_heads,
            'clamp_fraction': clamp_fraction,
            'noise_level': noise_level,
            'seed': seed
        },
        'training': {
            'final_acc': train_result['final_acc'],
            'history': train_result['history']
        },
        'natural_amplitude': float(natural_amplitude),
        'clamp_target': float(clamp_target),
        'conditions': []
    }
    
    # Test conditions: (clamp, noise)
    conditions = [
        ('baseline_clean', None, 0.0),
        ('baseline_noisy', None, noise_level),
        ('sedated_clean', clamp_target, 0.0),
        ('sedated_noisy', clamp_target, noise_level),
    ]
    
    for name, clamp, noise in conditions:
        metrics = evaluate_with_sedation(
            model, val_loader, clamp, noise, device, vocab_size
        )
        
        clamp_str = f"{clamp:.1f}" if clamp else "natural"
        noise_str = f"σ={noise}" if noise > 0 else "clean"
        
        print(f"{name:20s} | Clamp={clamp_str:8s} | Noise={noise_str:6s} | "
              f"Acc={metrics['accuracy']:.3f} | Margin={metrics['mean_margin']:.2f}")
        
        results['conditions'].append({
            'name': name,
            'clamp_target': float(clamp) if clamp else None,
            'noise_scale': float(noise),
            'accuracy': float(metrics['accuracy']),
            'mean_margin': float(metrics['mean_margin'])
        })
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80 + "\n")
    
    baseline_clean = results['conditions'][0]['accuracy']
    baseline_noisy = results['conditions'][1]['accuracy']
    sedated_clean = results['conditions'][2]['accuracy']
    sedated_noisy = results['conditions'][3]['accuracy']
    
    clean_degradation = (baseline_clean - sedated_clean) * 100
    noisy_degradation = (baseline_noisy - sedated_noisy) * 100
    
    print(f"Clean degradation:  {clean_degradation:.1f}%")
    print(f"Noisy degradation:  {noisy_degradation:.1f}%")
    
    # Success criteria
    crit_1 = clean_degradation < 5
    crit_2 = noisy_degradation > 20
    
    print("\n" + "-"*40)
    print("SUCCESS CRITERIA")
    print("-"*40 + "\n")
    
    print(f"1. Clean degradation <5%:   {'✓ PASS' if crit_1 else '✗ FAIL'} ({clean_degradation:.1f}%)")
    print(f"2. Noisy degradation >20%:  {'✓ PASS' if crit_2 else '✗ FAIL'} ({noisy_degradation:.1f}%)")
    
    overall = crit_1 and crit_2
    print(f"\nOverall: {'✓ MARGIN-AS-BUDGET HOLDS (2-LAYER)' if overall else '✗ INCONCLUSIVE'}")
    
    results['analysis'] = {
        'clean_degradation_pct': float(clean_degradation),
        'noisy_degradation_pct': float(noisy_degradation),
        'clean_pass': bool(crit_1),
        'noisy_pass': bool(crit_2),
        'overall_pass': bool(overall)
    }
    
    return results


def run_experiment(
    seq_len: int = 128,
    vocab_size: int = 4096,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    n_steps: int = 15000,
    batch_size: int = 64,
    lr: float = 1e-3,
    clamp_fraction: float = 0.6,
    noise_level: float = 2.0,
    start_seed: int = 42,
    n_seeds: int = 5,
    device: str = 'cuda'
):
    """Run full experiment with multiple seeds."""
    
    seeds = range(start_seed, start_seed + n_seeds)
    all_results = []
    
    print("\n" + "="*80)
    print(f"RUNNING 2-LAYER SEDATION EXPERIMENT (n={n_seeds} seeds)")
    print("="*80 + "\n")
    
    for i, seed in enumerate(seeds):
        print(f"\n>>> SEED {seed} ({i+1}/{len(seeds)})")
        res = run_single_seed(
            seq_len=seq_len, vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, n_steps=n_steps, batch_size=batch_size, lr=lr,
            clamp_fraction=clamp_fraction, noise_level=noise_level,
            seed=seed, device=device
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
    print("EXPERIMENT 2.3 RESULTS (Multi-seed)")
    print("="*80 + "\n")
    
    clean_deg = [r['analysis']['clean_degradation_pct'] for r in all_results]
    noisy_deg = [r['analysis']['noisy_degradation_pct'] for r in all_results]
    passes = [r['analysis']['overall_pass'] for r in all_results]
    
    m_clean, ci_clean = get_stats(clean_deg)
    m_noisy, ci_noisy = get_stats(noisy_deg)
    
    print(f"Clean Degradation:  {m_clean:.1f}% ± {ci_clean:.1f}%")
    print(f"Noisy Degradation:  {m_noisy:.1f}% ± {ci_noisy:.1f}%")
    print(f"Pass Rate:          {sum(passes)}/{len(passes)} ({(sum(passes)/len(passes)):.0%})")
    
    # Save
    save_path = Path("clean_audit/data/exp_2_3_sedation_deep_results.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {save_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Experiment 2.3: 2-Layer Sedation Test")
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=15000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clamp_fraction', type=float, default=0.6)
    parser.add_argument('--noise_level', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick_test', action='store_true')
    
    args = parser.parse_args()
    
    n_steps = 3000 if args.quick_test else args.n_steps
    n_seeds = 1 if args.quick_test else args.n_seeds
    
    if args.quick_test:
        print("\n[QUICK TEST MODE] Reduced steps and seeds\n")
    
    run_experiment(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_steps=n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        clamp_fraction=args.clamp_fraction,
        noise_level=args.noise_level,
        start_seed=args.seed,
        n_seeds=n_seeds,
        device=args.device
    )


if __name__ == '__main__':
    main()
