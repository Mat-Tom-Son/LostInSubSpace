"""
Generate Lambda Sweep Survival Curves

Compares survival curves across λ ∈ {0, 0.05, 0.3}
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    path = Path("data/exp_4_2c_lambda_sweep_results.json")
    with open(path) as f:
        return json.load(f)

def compute_survival_curve(trajectories, threshold=0.90):
    all_steps = set()
    for traj in trajectories:
        for step, acc in traj:
            all_steps.add(step)
    steps = sorted(all_steps)
    
    n_seeds = len(trajectories)
    stability_times = []
    
    for traj in trajectories:
        n = len(traj)
        stable_time = None
        for i in range(n):
            step, acc = traj[i]
            if acc >= threshold:
                stays_stable = all(traj[j][1] >= threshold for j in range(i + 1, n))
                if stays_stable:
                    stable_time = step
                    break
        stability_times.append(stable_time)
    
    survival = []
    for step in steps:
        not_stable = sum(1 for st in stability_times if st is None or st > step)
        survival.append((step, not_stable / n_seeds))
    
    return survival, stability_times

def plot_lambda_sweep(data, save_path="paper/lambda_sweep_survival.png"):
    """Generate comparative survival curves for λ sweep."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    lambdas = data['lambdas']
    results = data['results']
    
    colors = {'0.0': '#2ecc71', '0.05': '#3498db', '0.3': '#e74c3c'}
    labels = {'0.0': 'λ=0 (baseline)', '0.05': 'λ=0.05', '0.3': 'λ=0.3'}
    
    # --- Left: Survival curves ---
    ax1 = axes[0]
    
    for lam_str, color in colors.items():
        lam_results = results[lam_str]
        trajectories = [r['trajectory'] for r in lam_results]
        survival, _ = compute_survival_curve(trajectories)
        
        steps = [s for s, p in survival]
        probs = [p for s, p in survival]
        
        ax1.step(steps, probs, where='post', linewidth=2.5, 
                 label=labels[lam_str], color=color, alpha=0.9)
    
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('P(Not Yet Stable)', fontsize=12)
    ax1.set_title('Survival Curves by λ (4-Layer)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_xlim(0, 20000)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # --- Right: Summary bar chart ---
    ax2 = axes[1]
    
    summary = data['summary']
    x = np.arange(len(lambdas))
    width = 0.25
    
    stability_rates = [summary[str(lam)]['stability_rate'] * 100 for lam in lambdas]
    collapse_counts = [summary[str(lam)]['mean_collapses'] for lam in lambdas]
    
    bars1 = ax2.bar(x - width/2, stability_rates, width, label='Stability %', 
                    color='#27ae60', alpha=0.8)
    
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, collapse_counts, width, label='Collapses', 
                         color='#c0392b', alpha=0.8)
    
    ax2.set_xlabel('λ (Ortho Penalty)', fontsize=12)
    ax2.set_ylabel('Stability Rate (%)', fontsize=12, color='#27ae60')
    ax2_twin.set_ylabel('Mean Collapse Count', fontsize=12, color='#c0392b')
    ax2.set_title('λ Effect on Stability', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'λ={lam}' for lam in lambdas])
    ax2.tick_params(axis='y', labelcolor='#27ae60')
    ax2_twin.tick_params(axis='y', labelcolor='#c0392b')
    ax2.set_ylim(0, 60)
    ax2_twin.set_ylim(0, 10)
    
    # Add value labels on bars
    for bar, val in zip(bars1, stability_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{val:.0f}%', ha='center', va='bottom', fontsize=10, color='#27ae60')
    for bar, val in zip(bars2, collapse_counts):
        ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                      f'{val:.1f}', ha='center', va='bottom', fontsize=10, color='#c0392b')
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {save_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("LAMBDA SWEEP ANALYSIS")
    print("="*60)
    print("\nKey Finding: λ=0 (no ortho) has HIGHEST stability rate!")
    print("\n" + "-"*60)
    for lam in lambdas:
        s = summary[str(lam)]
        print(f"λ={lam}: {s['stability_rate']*100:.0f}% stable, "
              f"{s['mean_collapses']:.2f} collapses, "
              f"{s['mean_acc']:.3f} mean acc")
    
    plt.show()
    return fig

if __name__ == '__main__':
    data = load_results()
    plot_lambda_sweep(data)
