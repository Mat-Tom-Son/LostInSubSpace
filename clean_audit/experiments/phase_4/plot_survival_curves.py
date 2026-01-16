"""
Generate Survival Curves for Stability Characterization

Treats "stabilized by step t" as a survival event.
Creates Kaplan-Meier style curves for anchor vs probe.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    path = Path("clean_audit/data/exp_4_2b_stability_results.json")
    with open(path) as f:
        return json.load(f)

def compute_survival_curve(trajectories, threshold=0.90):
    """
    Compute survival curve: P(not yet stable at step t)
    
    A model is considered "stable from step t" if it never drops
    below threshold after step t.
    """
    # Get all unique steps
    all_steps = set()
    for traj in trajectories:
        for step, acc in traj:
            all_steps.add(step)
    steps = sorted(all_steps)
    
    n_seeds = len(trajectories)
    
    # For each seed, find time-to-stability
    stability_times = []
    for traj in trajectories:
        n = len(traj)
        stable_time = None
        for i in range(n):
            step, acc = traj[i]
            if acc >= threshold:
                # Check if we stay above threshold
                stays_stable = True
                for j in range(i + 1, n):
                    _, future_acc = traj[j]
                    if future_acc < threshold:
                        stays_stable = False
                        break
                if stays_stable:
                    stable_time = step
                    break
        stability_times.append(stable_time)
    
    # Build survival curve
    survival = []
    for step in steps:
        # Count seeds NOT YET stable at this step
        not_stable = 0
        for st in stability_times:
            if st is None or st > step:
                not_stable += 1
        survival.append((step, not_stable / n_seeds))
    
    return survival, stability_times

def plot_survival_curves(results, save_path="paper/survival_curves.png"):
    """Generate publication-ready survival curves."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Left: Survival curves ---
    ax1 = axes[0]
    
    anchor_trajectories = [r['anchor']['trajectory'] for r in results]
    probe_trajectories = [r['probe']['trajectory'] for r in results]
    
    anchor_survival, anchor_times = compute_survival_curve(anchor_trajectories)
    probe_survival, probe_times = compute_survival_curve(probe_trajectories)
    
    # Plot step-function style
    steps_a = [s for s, p in anchor_survival]
    probs_a = [p for s, p in anchor_survival]
    steps_p = [s for s, p in probe_survival]
    probs_p = [p for s, p in probe_survival]
    
    ax1.step(steps_a, probs_a, where='post', linewidth=2.5, 
             label='Anchor (no ortho)', color='#e74c3c', alpha=0.9)
    ax1.step(steps_p, probs_p, where='post', linewidth=2.5, 
             label='Probe (λ=0.5)', color='#3498db', alpha=0.9)
    
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('P(Not Yet Stable)', fontsize=12)
    ax1.set_title('Survival Curves: Time to Stability (4-Layer)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_xlim(0, 20000)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% unstable')
    
    # Add annotation
    ax1.annotate('Stochastic Escape\nProcess', xy=(10000, 0.7), fontsize=10, 
                 style='italic', color='gray')
    
    # --- Right: Collapse count comparison ---
    ax2 = axes[1]
    
    anchor_collapses = [r['anchor']['metrics']['collapse_count'] for r in results]
    probe_collapses = [r['probe']['metrics']['collapse_count'] for r in results]
    
    seeds = list(range(len(results)))
    x = np.arange(len(seeds))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, anchor_collapses, width, label='Anchor', color='#e74c3c', alpha=0.8)
    bars2 = ax2.bar(x + width/2, probe_collapses, width, label='Probe', color='#3498db', alpha=0.8)
    
    ax2.set_xlabel('Seed Index', fontsize=12)
    ax2.set_ylabel('Collapse Count', fontsize=12)
    ax2.set_title('Oscillatory Collapses per Seed', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Seed {i+1}' for i in seeds])
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add mean lines
    ax2.axhline(y=np.mean(anchor_collapses), color='#e74c3c', linestyle='--', alpha=0.7)
    ax2.axhline(y=np.mean(probe_collapses), color='#3498db', linestyle='--', alpha=0.7)
    
    # Annotate means
    ax2.annotate(f'μ={np.mean(anchor_collapses):.1f}', xy=(len(seeds)-0.5, np.mean(anchor_collapses)+0.3), 
                 color='#e74c3c', fontsize=10)
    ax2.annotate(f'μ={np.mean(probe_collapses):.1f}', xy=(len(seeds)-0.5, np.mean(probe_collapses)+0.3), 
                 color='#3498db', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {save_path}")
    
    # Also save summary stats
    print("\n" + "="*60)
    print("SURVIVAL ANALYSIS SUMMARY")
    print("="*60)
    
    anchor_stable = sum(1 for t in anchor_times if t is not None)
    probe_stable = sum(1 for t in probe_times if t is not None)
    
    print(f"\nStability Rate:")
    print(f"  Anchor: {anchor_stable}/{len(anchor_times)} ({100*anchor_stable/len(anchor_times):.0f}%)")
    print(f"  Probe:  {probe_stable}/{len(probe_times)} ({100*probe_stable/len(probe_times):.0f}%)")
    
    print(f"\nCollapse Count:")
    print(f"  Anchor: {np.mean(anchor_collapses):.2f} ± {np.std(anchor_collapses):.2f}")
    print(f"  Probe:  {np.mean(probe_collapses):.2f} ± {np.std(probe_collapses):.2f}")
    
    if anchor_stable > 0:
        anchor_tts = [t for t in anchor_times if t is not None]
        print(f"\nTime-to-Stability (when achieved):")
        print(f"  Anchor: {np.mean(anchor_tts):.0f} ± {np.std(anchor_tts):.0f}")
    if probe_stable > 0:
        probe_tts = [t for t in probe_times if t is not None]
        print(f"  Probe:  {np.mean(probe_tts):.0f} ± {np.std(probe_tts):.0f}")
    
    plt.show()
    return fig

if __name__ == '__main__':
    results = load_results()
    plot_survival_curves(results)
