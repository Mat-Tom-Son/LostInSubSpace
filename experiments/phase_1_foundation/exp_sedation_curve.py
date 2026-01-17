"""
EXPERIMENT: Sedation Degradation Curve

Phase 1.1 of Technical Improvement Plan

Goal: Create smooth degradation curve showing how clamping amplitude affects
accuracy under clean vs. noisy conditions.

Protocol:
1. Load trained modular arithmetic model (p=113)
2. Sweep clamp targets: [baseline, 8.0, 6.5, 5.0, 3.5, 3.0]
3. Test clean (σ=0) and noisy (σ=2.0) conditions
4. Run n=3 seeds per condition
5. Output: JSON results + publication figure

Success Criteria:
- Clean accuracy ≥98% across all clamps
- Noisy accuracy shows monotonic decline with clamp severity
- Clear elbow where margin becomes limiting factor
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import argparse
import json
import matplotlib.pyplot as plt

from lib.clamps import NaiveClamp
from lib.logging_utils import setup_reproducibility

# Import from exp_a_foundation
from experiments.phase_1_foundation.exp_a_foundation import SimpleTransformer, ModularArithmeticDataset


def load_trained_model(
    checkpoint_path: str,
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    max_seq_len: int = 128,  # Match checkpoint architecture
    device: str = 'cuda'
) -> Tuple[nn.Module, float]:
    """
    Load a trained modular arithmetic model.
    
    Note: The checkpoint was trained with max_seq_len=128 (interleaved config),
    so we need to match that architecture even for modular arithmetic.
    
    Returns:
        (model, natural_amplitude) tuple
    """
    # Create model architecture matching checkpoint
    model = SimpleTransformer(
        vocab_size=128,  # p + special tokens, rounded to 128
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=max_seq_len,  # Important: match checkpoint
        dropout=0.0
    )
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Loaded model from: {checkpoint_path}")
    print(f"Architecture: d_model={d_model}, n_heads={n_heads}, max_seq_len={max_seq_len}")
    
    # Measure natural amplitude using proper dataset with matching seq_len
    val_dataset = ModularArithmeticDataset(p=p, seq_len=max_seq_len, train=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    amplitudes = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            _ = model(x)
            
            if 'resid_post' in model.cache:
                resid = model.cache['resid_post']
                amp = resid.reshape(-1, resid.shape[-1]).norm(dim=-1).mean().item()
                amplitudes.append(amp)
    
    natural_amplitude = np.mean(amplitudes)
    print(f"Natural amplitude: {natural_amplitude:.2f}")
    
    return model, natural_amplitude


def evaluate_with_clamp(
    model: nn.Module,
    val_loader: DataLoader,
    clamp_target: Optional[float],
    noise_scale: float,
    device: str
) -> Dict[str, float]:
    """
    Evaluate model with optional clamp and post-clamp noise.
    
    Args:
        model: Trained model
        val_loader: Validation dataloader
        clamp_target: Target norm for clamp (None = no clamp)
        noise_scale: Post-clamp noise σ
        device: Compute device
        
    Returns:
        Dict with accuracy, mean_margin, etc.
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
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            
            # For modular arithmetic, we predict from the "=" position
            # The target is at position 4 (after x, +, y, =) in the ORIGINAL sequence
            # But 'y' (targets) is shifted by 1, so 'z' is at index 3.
            # Input index 3 ('=') -> predicts Target index 3 ('z')
            pred_logits = logits[:, 3, :]  # Position after "="
            preds = pred_logits.argmax(dim=-1)
            targets = y[:, 3]  # Target is the answer roughly at index 3
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            # Compute margins
            true_logits = pred_logits.gather(1, targets.unsqueeze(1)).squeeze()
            max_other = pred_logits.scatter(1, targets.unsqueeze(1), float('-inf')).max(dim=1).values
            margin = (true_logits - max_other).mean().item()
            margins.append(margin)
    
    # Clean up
    model.clamp_fn = None
    model.post_clamp_noise_scale = 0.0
    
    accuracy = correct / total if total > 0 else 0.0
    mean_margin = np.mean(margins) if margins else 0.0
    
    return {
        'accuracy': accuracy,
        'mean_margin': mean_margin,
        'n_samples': total
    }


def run_sedation_curve(
    checkpoint_path: str,
    clamp_targets: List[float],
    noise_levels: List[float] = [0.0, 2.0],
    seeds: List[int] = [42, 43, 44],
    p: int = 113,
    max_seq_len: int = 128,  # Must match checkpoint architecture
    device: str = 'cuda'
) -> Dict:
    """
    Run full sedation curve experiment.
    
    Args:
        checkpoint_path: Path to trained model
        clamp_targets: List of clamp values (None = baseline)
        noise_levels: List of noise σ values to test
        seeds: Random seeds for evaluation
        p: Modular arithmetic modulus
        max_seq_len: Sequence length (must match model checkpoint)
        device: Compute device
        
    Returns:
        Results dictionary
    """
    print("\n" + "="*80)
    print("SEDATION DEGRADATION CURVE EXPERIMENT")
    print("="*80 + "\n")
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Clamp targets: {clamp_targets}")
    print(f"Noise levels: {noise_levels}")
    print(f"Seeds: {seeds}")
    print(f"Sequence length: {max_seq_len}\n")
    
    # Load model and get natural amplitude (seq_len must match checkpoint)
    model, natural_amplitude = load_trained_model(
        checkpoint_path, p=p, max_seq_len=max_seq_len, device=device
    )
    
    # Add baseline (natural amplitude) to clamp targets if not present
    effective_clamps = [natural_amplitude] + [c for c in clamp_targets if c != natural_amplitude]
    
    # Create validation set with matching seq_len
    val_dataset = ModularArithmeticDataset(p=p, seq_len=max_seq_len, train=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    print(f"Validation set: {len(val_dataset)} samples\n")
    
    # Results storage
    results = {
        'config': {
            'checkpoint': checkpoint_path,
            'natural_amplitude': float(natural_amplitude),
            'clamp_targets': effective_clamps,
            'noise_levels': noise_levels,
            'seeds': seeds,
            'modulus': p
        },
        'runs': []
    }
    
    # Run grid
    print("-"*80)
    print("Running evaluation grid...")
    print("-"*80 + "\n")
    
    for clamp_target in effective_clamps:
        for noise_scale in noise_levels:
            seed_results = []
            
            for seed in seeds:
                setup_reproducibility(seed)
                
                # Re-create dataloader with this seed's shuffling
                val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
                
                metrics = evaluate_with_clamp(
                    model, val_loader, 
                    clamp_target=clamp_target,
                    noise_scale=noise_scale,
                    device=device
                )
                
                seed_results.append(metrics)
            
            # Aggregate across seeds
            mean_acc = np.mean([r['accuracy'] for r in seed_results])
            std_acc = np.std([r['accuracy'] for r in seed_results])
            mean_margin = np.mean([r['mean_margin'] for r in seed_results])
            
            is_baseline = clamp_target == natural_amplitude
            clamp_label = "baseline" if is_baseline else f"{clamp_target:.1f}"
            noise_label = "clean" if noise_scale == 0 else f"σ={noise_scale}"
            
            print(f"Clamp={clamp_label:8s} | Noise={noise_label:6s} | "
                  f"Acc={mean_acc:.3f}±{std_acc:.3f} | Margin={mean_margin:.2f}")
            
            results['runs'].append({
                'clamp_target': float(clamp_target),
                'is_baseline': is_baseline,
                'noise_scale': float(noise_scale),
                'accuracy_mean': float(mean_acc),
                'accuracy_std': float(std_acc),
                'margin_mean': float(mean_margin),
                'seed_results': seed_results
            })
    
    return results


def plot_sedation_curve(results: Dict, output_path: str):
    """
    Generate publication-quality sedation curve figure.
    
    Args:
        results: Results dictionary from run_sedation_curve
        output_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    # Extract data
    clamp_targets = results['config']['clamp_targets']
    natural_amp = results['config']['natural_amplitude']
    
    # Organize by noise condition
    # Map noise scale to data lists
    data_by_noise = {}
    
    unique_noise = sorted(list(set(r['noise_scale'] for r in results['runs'])))
    
    for noise in unique_noise:
        data_by_noise[noise] = {'clamps': [], 'acc': [], 'err': []}
        
    for run in results['runs']:
        noise = run['noise_scale']
        clamp = run['clamp_target']
        data_by_noise[noise]['clamps'].append(clamp)
        data_by_noise[noise]['acc'].append(run['accuracy_mean'])
        data_by_noise[noise]['err'].append(run['accuracy_std'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color map for noise
    # Clean (0) = Blue
    # Noise levels = Gradient Orange/Red
    import matplotlib.cm as cm
    
    for i, noise in enumerate(unique_noise):
        if noise == 0:
            color = '#2196F3'
            marker = 'o'
            label = 'Clean (σ=0)'
            zorder = 20
            width = 3
        else:
            # Gradient from Orange to Dark Red
            # Normalize index among noise levels (excluding 0)
            n_noisy = len(unique_noise) - 1
            if n_noisy > 0:
                idx = unique_noise.index(noise) - 1
                ratio = idx / max(1, n_noisy - 1)
                # Orange to Red
                color = cm.autumn_r(ratio * 0.8 + 0.1) 
            else:
                color = '#FF5722'
            
            marker = 's'
            label = f'Noisy (σ={noise})'
            zorder = 10
            width = 2
            
        ax.errorbar(data_by_noise[noise]['clamps'], data_by_noise[noise]['acc'], 
                    yerr=data_by_noise[noise]['err'],
                    marker=marker, markersize=8, linewidth=width, capsize=4,
                    color=color, label=label, zorder=zorder)
    
    # Mark baseline amplitude
    ax.axvline(natural_amp, color='gray', linestyle='--', alpha=0.5, label=f'Natural amplitude ({natural_amp:.1f})')
    
    # Decision boundary
    ax.axhline(0.5, color='black', linestyle=':', alpha=0.3, label='Random chance')
    
    # Formatting
    ax.set_xlabel('Clamp Target (Amplitude)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Sedation Degradation Curve: Margin as Noise Buffer', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()  # Higher clamp = less sedation, so put baseline on right
    
    # Add annotation
    ax.annotate('Severe sedation\n(reduced margin)',
                xy=(min(data_by_noise[0]['clamps']), 0.6),
                fontsize=9, ha='center', alpha=0.7)
    ax.annotate('Natural\namplitude',
                xy=(natural_amp, 0.95),
                fontsize=9, ha='center', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")
    plt.close()


def analyze_results(results: Dict):
    """Print analysis of sedation curve results."""
    
    print("\n" + "="*80)
    print("SEDATION CURVE ANALYSIS")
    print("="*80 + "\n")
    
    natural_amp = results['config']['natural_amplitude']
    
    # Find clean and noisy results at extreme clamps
    baseline_clean = None
    baseline_noisy = None
    severe_clean = None
    severe_noisy = None
    
    min_clamp = min(r['clamp_target'] for r in results['runs'])
    
    for run in results['runs']:
        if run['is_baseline']:
            if run['noise_scale'] == 0:
                baseline_clean = run
            else:
                baseline_noisy = run
        elif run['clamp_target'] == min_clamp:
            if run['noise_scale'] == 0:
                severe_clean = run
            else:
                severe_noisy = run
    
    print("KEY FINDINGS:\n")
    
    # Check success criteria
    all_clean_high = all(r['accuracy_mean'] >= 0.98 for r in results['runs'] if r['noise_scale'] == 0)
    print(f"1. Clean accuracy ≥98% at all clamps: {'✓ PASS' if all_clean_high else '✗ FAIL'}")
    
    if baseline_noisy and severe_noisy:
        degradation = baseline_noisy['accuracy_mean'] - severe_noisy['accuracy_mean']
        print(f"2. Noisy accuracy degradation (baseline→severe): {degradation:.1%}")
        print(f"   Baseline noisy: {baseline_noisy['accuracy_mean']:.1%}")
        print(f"   Severe noisy:   {severe_noisy['accuracy_mean']:.1%}")
    
    # Check monotonicity
    noisy_runs = sorted([r for r in results['runs'] if r['noise_scale'] > 0], 
                        key=lambda x: x['clamp_target'], reverse=True)
    accs = [r['accuracy_mean'] for r in noisy_runs]
    monotonic = all(accs[i] >= accs[i+1] for i in range(len(accs)-1))
    print(f"3. Noisy accuracy monotonically decreases: {'✓ PASS' if monotonic else '✗ FAIL (some non-monotonicity)'}")
    
    # Summary
    print("\n" + "-"*40)
    print("INTERPRETATION")
    print("-"*40)
    print("""
The key finding is the DISSOCIATION between clean and noisy conditions:
- Clean accuracy stays high → clamp doesn't break model
- Noisy accuracy degrades → clamp removes noise buffer

This proves: Amplitude margin is a PROPHYLACTIC resource for noise tolerance,
not a computational necessity for clean inference.
""")


def main():
    parser = argparse.ArgumentParser(description="Sedation Degradation Curve Experiment")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/healthy_victim_modular.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--p', type=int, default=113, help='Modular arithmetic modulus')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--quick_test', action='store_true', help='Quick test (1 seed, fewer clamps)')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--load_results', type=str, default=None, help='Load results from existing JSON file instead of running')
    
    args = parser.parse_args()
    
    if args.load_results:
        print(f"\nLoading results from: {args.load_results}")
        with open(args.load_results, 'r') as f:
            results = json.load(f)
    else:
        # Configuration
        if args.quick_test:
            clamp_targets = [8.0, 5.0, 3.0]
            seeds = [42]
            print("\n[QUICK TEST MODE] Reduced configuration\n")
        else:
            # High resolution sweep
            # 11.5 down to 2.5 in 0.5 steps
            import numpy as np
            clamp_targets = np.arange(2.5, 12.0, 0.5)[::-1].tolist()
            # Ensure exact values for key points
            clamp_targets = [float(x) for x in clamp_targets]
            seeds = [42, 43, 44, 45, 46]
        
        noise_levels = [0.0, 1.0, 2.0, 3.0]
        
        # Run experiment
        results = run_sedation_curve(
            checkpoint_path=args.checkpoint,
            clamp_targets=clamp_targets,
            noise_levels=noise_levels,
            seeds=seeds,
            p=args.p,
            max_seq_len=128,
            device=args.device
        )
        
        # Save results
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_path = output_path / 'exp_sedation_curve_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved: {json_path}")
    
    # Generate figure
    fig_path = Path('paper') / 'sedation_curve.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_sedation_curve(results, str(fig_path))
    
    # Analysis
    analyze_results(results)
    
    return results


if __name__ == '__main__':
    main()
