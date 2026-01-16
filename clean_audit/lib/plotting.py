"""
Visualization Utilities for Allostatic Load Research

Generates publication-quality figures for all experiments:
  - Figure 1 (Exp A): Overlay plot (Ψ, G, A, σ²) vs training step
  - Figure 2 (Exp B): Heatmap of residual norm ratios by layer
  - Figure 3 (Exp C): Phase diagram (Accuracy vs A_param)
  - Figure 4 (Exp C): Time series showing divergence
  - Figure 5 (Exp D): Box plots of variance by site type
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


# Set publication style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


def plot_overlay_time_series(
    log_files: Dict[str, str],
    metrics: List[str],
    output_path: str,
    title: str = "Experiment A: Foundation (Constraint + Clamp)",
    y_labels: Optional[Dict[str, str]] = None
):
    """
    Figure 1: Overlay plot of multiple metrics across conditions.

    Args:
        log_files: Dict of condition_name -> log_filepath
        metrics: List of metric names to plot
        output_path: Where to save figure
        title: Plot title
        y_labels: Optional custom y-axis labels for metrics
    """
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))

    for ax, metric in zip(axes, metrics):
        for (condition_name, log_file), color in zip(log_files.items(), colors):
            # Load time series
            with open(log_file, 'r') as f:
                log = json.load(f)

            steps = []
            values = []
            for record in log['metrics']:
                if metric in record:
                    steps.append(record['step'])
                    values.append(record[metric])

            # Plot
            ax.plot(steps, values, label=condition_name, color=color, linewidth=2, alpha=0.8)

        # Format
        y_label = y_labels.get(metric, metric) if y_labels else metric
        ax.set_ylabel(y_label, fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Training Step', fontsize=14)
    fig.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    plt.close()


def plot_heatmap_layer_compensation(
    compensation_by_layer: Dict[int, float],
    ablation_layer: int,
    output_path: str,
    title: str = "Experiment B: Compensation Heatmap"
):
    """
    Figure 2: Heatmap of residual norm ratios by layer.

    Args:
        compensation_by_layer: Dict of layer_idx -> compensation_ratio
        ablation_layer: Which layer was ablated
        output_path: Where to save figure
        title: Plot title
    """
    layers = sorted(compensation_by_layer.keys())
    ratios = [compensation_by_layer[l] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create heatmap data
    data = np.array(ratios).reshape(1, -1)

    # Plot
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.5)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_yticks([0])
    ax.set_yticklabels(['Norm Ratio'])
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_title(title, fontsize=16)

    # Mark ablation site
    ax.axvline(ablation_layer - 0.5, color='red', linewidth=3, linestyle='--', alpha=0.7)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Residual Norm (Ablated / Baseline)', rotation=270, labelpad=20)

    # Add reference line at 1.0
    ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)

    # Annotate values
    for i, (layer, ratio) in enumerate(zip(layers, ratios)):
        color = 'white' if 0.7 < ratio < 1.3 else 'black'
        ax.text(i, 0, f"{ratio:.2f}", ha='center', va='center', color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    plt.close()


def plot_phase_diagram(
    log_files: Dict[str, str],
    x_metric: str = "A_param",
    y_metric: str = "psi_accuracy",
    output_path: str,
    title: str = "Experiment C: Phase Diagram (Grokking)"
):
    """
    Figure 3: Phase diagram showing trajectory in (A_param, Accuracy) space.

    Args:
        log_files: Dict of condition_name -> log_filepath
        x_metric: Metric for x-axis (typically A_param)
        y_metric: Metric for y-axis (typically accuracy)
        output_path: Where to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))

    for (condition_name, log_file), color in zip(log_files.items(), colors):
        with open(log_file, 'r') as f:
            log = json.load(f)

        x_values = []
        y_values = []
        steps = []

        for record in log['metrics']:
            if x_metric in record and y_metric in record:
                x_values.append(record[x_metric])
                y_values.append(record[y_metric])
                steps.append(record['step'])

        if not x_values:
            continue

        # Plot trajectory
        ax.plot(x_values, y_values, color=color, alpha=0.6, linewidth=2, label=condition_name)

        # Mark start and end
        ax.scatter(x_values[0], y_values[0], color=color, marker='o', s=200,
                   edgecolor='black', linewidth=2, zorder=10, label=f'{condition_name} (start)')
        ax.scatter(x_values[-1], y_values[-1], color=color, marker='*', s=400,
                   edgecolor='black', linewidth=2, zorder=10, label=f'{condition_name} (end)')

        # Add arrows to show direction
        n_arrows = min(5, len(x_values) // 10)
        arrow_indices = np.linspace(10, len(x_values) - 10, n_arrows, dtype=int)
        for idx in arrow_indices:
            ax.annotate('', xy=(x_values[idx], y_values[idx]),
                       xytext=(x_values[idx-5], y_values[idx-5]),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6))

    ax.set_xlabel(x_metric, fontsize=14)
    ax.set_ylabel(y_metric, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    plt.close()


def plot_time_series_divergence(
    log_files: Dict[str, str],
    metrics: List[str],
    output_path: str,
    title: str = "Experiment C: Time Series (Weight Decay Effect)",
    highlight_transition: bool = True
):
    """
    Figure 4: Time series showing metric divergence between conditions.

    Args:
        log_files: Dict of condition_name -> log_filepath
        metrics: List of metrics to plot
        output_path: Where to save figure
        title: Plot title
        highlight_transition: Shade transition zone (15-85% accuracy)
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))

    for ax, metric in zip(axes, metrics):
        for (condition_name, log_file), color in zip(log_files.items(), colors):
            with open(log_file, 'r') as f:
                log = json.load(f)

            steps = []
            values = []
            accuracies = []

            for record in log['metrics']:
                if metric in record:
                    steps.append(record['step'])
                    values.append(record[metric])
                    accuracies.append(record.get('psi_accuracy', None))

            # Plot line
            ax.plot(steps, values, label=condition_name, color=color, linewidth=2.5, alpha=0.85)

            # Highlight transition zone if requested and we have accuracy data
            if highlight_transition and metric == 'psi_accuracy':
                transition_steps = [s for s, a in zip(steps, accuracies) if a is not None and 0.15 < a < 0.85]
                if transition_steps:
                    ax.axvspan(min(transition_steps), max(transition_steps),
                              alpha=0.2, color=color, label=f'{condition_name} transition')

        ax.set_ylabel(metric, fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Training Step', fontsize=14)
    fig.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    plt.close()


def plot_box_plots_variance(
    variance_by_site: Dict[str, List[float]],
    output_path: str,
    title: str = "Experiment D: Variance by Site Type"
):
    """
    Figure 5: Box plots comparing variance across site types.

    Args:
        variance_by_site: Dict of site_type -> [variance_values]
        output_path: Where to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data
    site_types = list(variance_by_site.keys())
    variances = [variance_by_site[site] for site in site_types]

    # Create box plot
    bp = ax.boxplot(variances, labels=site_types, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=10))

    # Color boxes
    colors = sns.color_palette("Set2", len(site_types))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add statistical annotations
    means = [np.mean(v) for v in variances]
    stds = [np.std(v) for v in variances]

    for i, (site, mean, std) in enumerate(zip(site_types, means, stds)):
        ax.text(i + 1, mean, f'μ={mean:.3f}\nσ={std:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_ylabel('Variance (σ²)', fontsize=14)
    ax.set_xlabel('Site Type', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')

    # Add significance markers if suppressors have >2x variance
    if 'suppressor' in variance_by_site and 'clean_early' in variance_by_site:
        suppressor_mean = np.mean(variance_by_site['suppressor'])
        clean_mean = np.mean(variance_by_site['clean_early'])
        ratio = suppressor_mean / clean_mean

        if ratio > 2.0:
            ax.text(0.5, 0.95, f'Suppressor/Clean ratio: {ratio:.2f}× (>2.0 threshold)',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=12, fontweight='bold', color='green',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    plt.close()


def plot_correlation_matrix(
    log_file: str,
    metrics: List[str],
    output_path: str,
    title: str = "Metric Correlation Matrix"
):
    """
    Plot correlation matrix between metrics.

    Useful for exploring relationships between Ψ, G, A, σ².

    Args:
        log_file: Path to audit log
        metrics: List of metrics to correlate
        output_path: Where to save figure
        title: Plot title
    """
    with open(log_file, 'r') as f:
        log = json.load(f)

    # Build data matrix
    data = {metric: [] for metric in metrics}

    for record in log['metrics']:
        for metric in metrics:
            if metric in record:
                data[metric].append(record[metric])
            else:
                data[metric].append(np.nan)

    # Convert to numpy and compute correlation
    import pandas as pd
    df = pd.DataFrame(data)
    corr_matrix = df.corr()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, vmin=-1, vmax=1, square=True,
               linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    plt.close()


def plot_scatter_with_regression(
    log_file: str,
    x_metric: str,
    y_metric: str,
    output_path: str,
    title: Optional[str] = None
):
    """
    Scatter plot with regression line for two metrics.

    Useful for testing Ψ = G + A hypothesis.

    Args:
        log_file: Path to audit log
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        output_path: Where to save figure
        title: Plot title (auto-generated if None)
    """
    with open(log_file, 'r') as f:
        log = json.load(f)

    x_values = []
    y_values = []

    for record in log['metrics']:
        if x_metric in record and y_metric in record:
            x_values.append(record[x_metric])
            y_values.append(record[y_metric])

    if not x_values:
        print(f"Warning: No data found for {x_metric} vs {y_metric}")
        return

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Compute regression
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_values, y_values, alpha=0.5, s=20)
    ax.plot(x_values, slope * x_values + intercept, 'r-', linewidth=2,
           label=f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}, p = {p_value:.2e}')

    ax.set_xlabel(x_metric, fontsize=14)
    ax.set_ylabel(y_metric, fontsize=14)
    if title is None:
        title = f'{y_metric} vs {x_metric}'
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', framealpha=0.9, fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    plt.close()


def create_figure_set_exp_a(
    log_dir: str,
    output_dir: str,
    conditions: List[str] = ['control', 'constraint', 'naive_clamp', 'mean_preserving_clamp']
):
    """
    Generate all figures for Experiment A.

    Args:
        log_dir: Directory containing audit logs
        output_dir: Where to save figures
        conditions: List of condition names
    """
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build log file dict
    log_files = {}
    for condition in conditions:
        log_path = log_dir / f"audit_log_exp_a_{condition}_seed_42.json"
        if log_path.exists():
            log_files[condition] = str(log_path)

    if not log_files:
        print("Warning: No log files found for Experiment A")
        return

    # Figure 1: Overlay plot
    metrics = ['psi_accuracy', 'A_activation_L0', 'sigma_sq_L0']
    y_labels = {
        'psi_accuracy': 'Ψ (Accuracy)',
        'A_activation_L0': 'A (Amplitude)',
        'sigma_sq_L0': 'σ² (Variance)'
    }

    plot_overlay_time_series(
        log_files,
        metrics,
        str(output_dir / "figure_1_exp_a_overlay.png"),
        title="Experiment A: Foundation (Constraint + Clamp)",
        y_labels=y_labels
    )

    print("Experiment A figures complete")


def create_all_figures(log_dir: str, output_dir: str):
    """
    Generate all figures for all experiments.

    Args:
        log_dir: Directory containing audit logs
        output_dir: Where to save figures
    """
    print("Generating all figures...")

    # Exp A
    create_figure_set_exp_a(log_dir, output_dir)

    # Additional experiments would be added here

    print(f"All figures saved to: {output_dir}")
