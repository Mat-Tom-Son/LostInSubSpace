"""
PHASE 2.2 ANALYSIS SCRIPT
Automated validation against PHASE_2_DECISION_TABLE.md

Run this immediately after experiments complete.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Noise levels tested
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20]
CONDITIONS = ['control', 'constraint', 'naive_clamp', 'mean_preserving_clamp']

# Hard thresholds from decision table
BASELINE_A_ACT = 11.47  # From Phase 1.1 control
SPIKE_THRESHOLD = BASELINE_A_ACT * 1.5  # 17.25
COLLAPSE_THRESHOLD = 0.20  # 20% accuracy
DIVERGENCE_THRESHOLD = 0.10  # 10 percentage points

DATA_DIR = Path('clean_audit/data')


def load_experiment_data(condition: str, noise: float) -> Dict:
    """Load metrics from experiment log."""
    # Logs are named: audit_log_exp_a_{condition}_seed_42.json
    # But we need to handle noise levels - they might be in separate runs
    # For now, assume we need to check file timestamps or naming convention

    fname = DATA_DIR / f"audit_log_exp_a_{condition}_seed_42_noise_{noise:.2f}.json"
    if not fname.exists():
        # Fallback: try without noise suffix (for noise=0.00 baseline)
        fname = DATA_DIR / f"audit_log_exp_a_{condition}_seed_42.json"

    if not fname.exists():
        return None

    with open(fname, 'r') as f:
        data = json.load(f)

    return data


def extract_final_metrics(data: Dict) -> Dict:
    """Extract final epoch metrics."""
    if data is None or 'metrics' not in data or len(data['metrics']) == 0:
        return None

    final = data['metrics'][-1]
    return {
        'val_acc': final.get('val_acc', float('nan')),
        'A_activation': final.get('A_activation', float('nan')),
        'sigma_sq': final.get('sigma_sq', float('nan')),
        'var_ratio_ff_attn': final.get('var_ratio_ff_attn', float('nan')),
        'snr_db': final.get('snr_db', float('nan'))
    }


def extract_trajectory(data: Dict, metric: str) -> Tuple[List[int], List[float]]:
    """Extract metric trajectory over epochs."""
    if data is None or 'metrics' not in data:
        return [], []

    epochs = []
    values = []

    for record in data['metrics']:
        step = record.get('step', 0)
        # Approximate epoch from step (assumes consistent batches/epoch)
        epoch = step // 157  # 10000 samples / 64 batch = ~157 batches/epoch
        value = record.get(metric, float('nan'))

        epochs.append(epoch)
        values.append(value)

    return epochs, values


def check_amplitude_spike(noise: float) -> Tuple[bool, float]:
    """Check if amplitude spiked for given noise level."""
    data = load_experiment_data('constraint', noise)
    if data is None:
        return False, float('nan')

    epochs, a_acts = extract_trajectory(data, 'A_activation')
    if len(a_acts) == 0:
        return False, float('nan')

    max_a_act = max(a_acts)
    spiked = max_a_act >= SPIKE_THRESHOLD

    return spiked, max_a_act


def check_clamp_divergence(noise: float) -> Tuple[bool, float]:
    """Check if Naive Clamp diverged from Constraint."""
    constraint_data = load_experiment_data('constraint', noise)
    clamp_data = load_experiment_data('naive_clamp', noise)

    if constraint_data is None or clamp_data is None:
        return False, float('nan')

    constraint_metrics = extract_final_metrics(constraint_data)
    clamp_metrics = extract_final_metrics(clamp_data)

    if constraint_metrics is None or clamp_metrics is None:
        return False, float('nan')

    gap = constraint_metrics['val_acc'] - clamp_metrics['val_acc']
    diverged = gap >= DIVERGENCE_THRESHOLD

    return diverged, gap


def classify_outcome() -> str:
    """Classify results into Outcome A, B, or C."""
    # Check critical noise levels (0.15 and 0.20)
    spike_015, max_a_015 = check_amplitude_spike(0.15)
    spike_020, max_a_020 = check_amplitude_spike(0.20)

    div_015, gap_015 = check_clamp_divergence(0.15)
    div_020, gap_020 = check_clamp_divergence(0.20)

    print("\n" + "="*80)
    print("OUTCOME CLASSIFICATION")
    print("="*80 + "\n")

    print("Critical Signal Check:")
    print(f"  Noise 0.15: A_act max = {max_a_015:.2f} (spike: {spike_015})")
    print(f"  Noise 0.20: A_act max = {max_a_020:.2f} (spike: {spike_020})")
    print(f"  Noise 0.15: Clamp gap = {gap_015:.3f} (diverged: {div_015})")
    print(f"  Noise 0.20: Clamp gap = {gap_020:.3f} (diverged: {div_020})")
    print()

    # Decision logic
    if (spike_015 or spike_020) and (div_015 or div_020):
        outcome = "A"
        interpretation = "Regime III Observed"
    elif not (spike_015 or spike_020) and not (div_015 or div_020):
        outcome = "B"
        interpretation = "Geometry Always Wins"
    else:
        outcome = "C"
        interpretation = "Narrow Window"

    print(f"OUTCOME: {outcome} - {interpretation}")
    print()

    return outcome


def plot_comparison_1():
    """A_activation trajectory across noise levels."""
    plt.figure(figsize=(10, 6))

    for noise in NOISE_LEVELS:
        data = load_experiment_data('constraint', noise)
        if data is None:
            continue

        epochs, a_acts = extract_trajectory(data, 'A_activation')
        plt.plot(epochs, a_acts, label=f'Noise {noise:.2f}', marker='o')

    plt.axhline(BASELINE_A_ACT, color='gray', linestyle='--', label='Baseline')
    plt.axhline(SPIKE_THRESHOLD, color='red', linestyle='--', label='Spike Threshold')

    plt.xlabel('Epoch')
    plt.ylabel('A_activation')
    plt.title('Comparison 1: A_activation Trajectory (Constraint Condition)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase2_comparison1_a_activation.png', dpi=150)
    print("Saved: phase2_comparison1_a_activation.png")


def plot_comparison_2():
    """Constraint vs Naive Clamp accuracy gap across noise levels."""
    gaps = []

    for noise in NOISE_LEVELS:
        diverged, gap = check_clamp_divergence(noise)
        gaps.append(gap)

    plt.figure(figsize=(8, 6))
    plt.plot(NOISE_LEVELS, gaps, marker='o', linewidth=2, markersize=8)
    plt.axhline(DIVERGENCE_THRESHOLD, color='red', linestyle='--', label='Divergence Threshold')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.3)

    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy Gap (Constraint - Naive Clamp)')
    plt.title('Comparison 2: Clamp Divergence vs Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase2_comparison2_clamp_divergence.png', dpi=150)
    print("Saved: phase2_comparison2_clamp_divergence.png")


def plot_comparison_3():
    """FFN/Attn variance ratio for critical noise 0.15."""
    data = load_experiment_data('constraint', 0.15)
    if data is None:
        print("Warning: No data for noise 0.15, skipping Comparison 3")
        return

    epochs, ratios = extract_trajectory(data, 'var_ratio_ff_attn')

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ratios, marker='o', color='purple')
    plt.axhline(3.73, color='gray', linestyle='--', label='Phase 1.1 Baseline (3.73)')

    plt.xlabel('Epoch')
    plt.ylabel('FFN/Attn Variance Ratio')
    plt.title('Comparison 3: Geometric Bypass Under Noise (0.15)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase2_comparison3_ffn_variance.png', dpi=150)
    print("Saved: phase2_comparison3_ffn_variance.png")


def print_results_table():
    """Print final results in tabular format."""
    print("\n" + "="*80)
    print("PHASE 2.2 FINAL RESULTS")
    print("="*80 + "\n")

    print(f"{'Noise':<8} {'Condition':<20} {'Accuracy':<10} {'A_act':<8} {'FFN/Attn':<8} {'SNR(dB)':<8}")
    print("-" * 80)

    for noise in NOISE_LEVELS:
        for condition in CONDITIONS:
            data = load_experiment_data(condition, noise)
            metrics = extract_final_metrics(data)

            if metrics is None:
                continue

            acc = metrics['val_acc']
            a_act = metrics['A_activation']
            ratio = metrics['var_ratio_ff_attn']
            snr = metrics['snr_db']

            print(f"{noise:<8.2f} {condition:<20} {acc:<10.4f} {a_act:<8.2f} {ratio:<8.2f} {snr:<8.1f}")

    print()


def main():
    """Run complete Phase 2.2 analysis."""
    print("\n" + "="*80)
    print("PHASE 2.2 ANALYSIS")
    print("="*80 + "\n")

    # Print results table
    print_results_table()

    # Classify outcome
    outcome = classify_outcome()

    # Generate plots
    print("Generating comparison plots...")
    plot_comparison_1()
    plot_comparison_2()
    plot_comparison_3()

    print("\nAnalysis complete.")
    print(f"Outcome: {outcome}")
    print("See PHASE_2_DECISION_TABLE.md for interpretation.")


if __name__ == '__main__':
    main()
