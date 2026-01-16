# The Conservation of Separability (v2.1)

**Project Code:** ALLOSTATIC_AUDIT
**Status:** Ready for Implementation
**Environment:** Clean Room (TransformerLens, ARENA, Public Data)

## Executive Summary

This repository contains the complete implementation of four controlled experiments designed to validate the **Conservation of Separability** hypothesis:

```
Ψ = G + A
```

Where:
- **Ψ (Separability)**: Functional performance (accuracy, logit_diff)
- **G (Geometry)**: Routing plasticity, directional alignment (low-energy)
- **A (Amplitude)**: Mean-shift magnitude in signal direction (high-energy)

**Core Hypothesis:** When G is constrained (frozen routing, architectural bottlenecks), the system must increase A to maintain Ψ. The "noise" visible in damaged models is **Allostatic Load**—the metabolic cost of amplitude-based compensation.

---

## Project Structure

```
clean_audit/
├── experiments/
│   ├── exp_a_foundation.py          # Constraint + Clamp experiments
│   ├── exp_b_repair.py              # GPT-2 Ablation audit
│   ├── exp_c_grokking.py            # Modular Addition + Weight Decay
│   └── exp_d_superposition.py       # Variance Analysis / Polysemanticity
├── lib/
│   ├── metrics.py                   # Unified observable definitions
│   ├── clamps.py                    # Naive + Mean-Preserving clamps
│   ├── logging_utils.py             # Adaptive frequency handler
│   └── plotting.py                  # Phase diagrams, overlays
├── data/
│   └── audit_log_*.json             # Structured metric output
└── README.md                        # This file
```

---

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended but not required)
- 16GB+ RAM

### Setup

```bash
# Clone or navigate to repository
cd allo-audit

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

- PyTorch >= 2.0.0
- TransformerLens >= 1.17.0
- NumPy, SciPy, Matplotlib, Seaborn
- Datasets, Transformers (HuggingFace)

---

## Experiments Overview

### Experiment A: Foundation (Constraint + Clamp)

**Goal:** Prove variance amplification is causal necessity under constraint.

**Conditions:**
1. **Control:** All 4 attention heads free (baseline)
2. **Constraint:** 3/4 heads frozen, 1 free (stress condition)
3. **Naive Clamp:** Constraint + clamp both A and σ² → catastrophic collapse
4. **Mean-Preserving Clamp:** Constraint + clamp σ² only → performance preserved

**Expected Results:**
- Control: ≥95% accuracy
- Constraint: 55-65% accuracy, A_activation 6x+ higher
- Naive Clamp: <5% accuracy (proves both A and σ² needed)
- Mean-Preserving: 50-54% accuracy (proves A is mechanism, σ² is byproduct)

**Run:**
```bash
# Quick test (10 epochs)
python clean_audit/experiments/exp_a_foundation.py --quick_test

# Full run (100 epochs)
python clean_audit/experiments/exp_a_foundation.py

# Specific condition
python clean_audit/experiments/exp_a_foundation.py --condition control --n_epochs 100
```

### Experiment B: Causal Audit (Ablation + Immediate Compensation)

**Goal:** Prove "self-repair" is "amplitude compensation."

**Resource:** GPT-2 Small (pre-trained), IOI task

**Conditions:**
1. **Baseline:** No ablation
2. **Critical Ablation:** Ablate name mover heads (L9H9, L9H6, L10H0)
3. **Random Control:** Ablate random head (L2H3)

**Expected Results:**
- Critical: ΔLogit_diff = -3.5, Compensation > 1.3× in layers 10-12
- Random: ΔLogit_diff ≈ -0.1, Compensation ≈ 1.0×

**Run:**
```bash
# Quick test
python clean_audit/experiments/exp_b_repair.py --quick_test

# Full run
python clean_audit/experiments/exp_b_repair.py --n_examples 100

# Specific condition
python clean_audit/experiments/exp_b_repair.py --condition critical
```

### Experiment C: Temporal Audit (Grokking + Weight Decay)

**Goal:** Reinterpret grokking as phase transition from A → G.

**Resource:** ARENA 3.0 modular addition (p=113)

**Conditions:**
1. **High Weight Decay (WD=1.0):** Phase transition A → G
2. **No Weight Decay (WD=0.0):** A persists, no transition

**Expected Results:**
- High WD: Phase transition at ~10k steps, ρ(acc, A_param) < -0.3
- No WD: A_param stays high, no sharp transition

**Run:**
```bash
# Quick test (5k steps)
python clean_audit/experiments/exp_c_grokking.py --quick_test

# Full run (50k steps)
python clean_audit/experiments/exp_c_grokking.py

# Specific condition
python clean_audit/experiments/exp_c_grokking.py --condition high_wd --n_steps 50000
```

### Experiment D: Structural Audit (Superposition)

**Goal:** Link polysemanticity to high-A regime.

**Method:** Functional suppressor identification via ablation (NOT circular)

**Expected Results:**
- Suppressor variance > 2.0× clean variance
- Bootstrap CI excludes 1.0 with 95% confidence

**Run:**
```bash
# Basic run
python clean_audit/experiments/exp_d_superposition.py

# Custom parameters
python clean_audit/experiments/exp_d_superposition.py --n_samples 2000 --train_steps 200
```

---

## Reproducibility

### Setting Seeds

All experiments support `--seed` argument for reproducibility:

```bash
python clean_audit/experiments/exp_a_foundation.py --seed 42
python clean_audit/experiments/exp_b_repair.py --seed 123
python clean_audit/experiments/exp_c_grokking.py --seed 2024
python clean_audit/experiments/exp_d_superposition.py --seed 777
```

### Output Logs

All experiments save structured JSON logs to `clean_audit/data/`:

```
audit_log_exp_a_control_seed_42.json
audit_log_exp_a_constraint_seed_42.json
audit_log_exp_b_seed_42.json
audit_log_exp_c_high_wd_seed_42.json
audit_log_exp_d_superposition_seed_42.json
```

### Log Format

```json
{
  "metadata": {
    "experiment": "exp_a_control",
    "seed": 42,
    "start_time": "2024-01-15T10:30:00",
    "git_commit": "abc123..."
  },
  "metrics": [
    {
      "step": 0,
      "timestamp": "2024-01-15T10:30:01",
      "psi_accuracy": 0.523,
      "A_activation": 0.342,
      "A_learned": 1.001,
      "A_param": 145.23,
      "sigma_sq": 0.089
    },
    ...
  ],
  "checkpoints": {
    "1000": "sha256:abc...",
    "2000": "sha256:def..."
  },
  "summary": {
    "total_steps": 5000,
    "duration_seconds": 3245.2
  }
}
```

---

## Visualization

### Generating Figures

Use the plotting utilities to generate publication-quality figures:

```python
from lib.plotting import (
    plot_overlay_time_series,
    plot_heatmap_layer_compensation,
    plot_phase_diagram,
    plot_time_series_divergence,
    plot_box_plots_variance,
    create_all_figures
)

# Generate all figures for all experiments
create_all_figures(
    log_dir="clean_audit/data",
    output_dir="clean_audit/figures"
)
```

### Individual Figures

**Figure 1 (Exp A):** Overlay plot (Ψ, A, σ²) vs training step
```python
plot_overlay_time_series(
    log_files={
        'control': 'clean_audit/data/audit_log_exp_a_control_seed_42.json',
        'constraint': 'clean_audit/data/audit_log_exp_a_constraint_seed_42.json',
        'naive_clamp': 'clean_audit/data/audit_log_exp_a_naive_clamp_seed_42.json',
        'mean_preserving': 'clean_audit/data/audit_log_exp_a_mean_preserving_clamp_seed_42.json'
    },
    metrics=['psi_accuracy', 'A_activation_L0', 'sigma_sq_L0'],
    output_path='figures/figure_1_exp_a.png'
)
```

**Figure 3 (Exp C):** Phase diagram (Accuracy vs A_param)
```python
plot_phase_diagram(
    log_files={
        'High WD': 'clean_audit/data/audit_log_exp_c_high_wd_seed_42.json',
        'No WD': 'clean_audit/data/audit_log_exp_c_no_wd_seed_42.json'
    },
    output_path='figures/figure_3_phase_diagram.png'
)
```

---

## Metrics Reference

### Ψ (Separability)
- **Definition:** Functional performance
- **Primary:** `psi_accuracy` (classification accuracy)
- **Alternate:** `psi_logit_diff` (for IOI task)

### G (Geometry / Routing Plasticity)
- **Definition:** 1 - (attention pattern drift over time)
- **Metric:** JSD-based, range [0, 1]
- **High G:** Routing still exploring (plastic)
- **Low G:** Routing frozen

### A (Amplitude)
Three variants:
1. **A_learned:** Mean scale parameter of final LayerNorm (`ln_final.w`)
2. **A_activation:** Global mean residual norm (per layer)
3. **A_param:** L2 norm of Value projection weights (W_V)

### σ² (Variance)
- **Definition:** Variance of residual activations (post-centering)
- **Interpretation:** Byproduct of amplitude scaling, NOT the mechanism

---

## Success Criteria

### Experiment A
- ✓ Control ≥95% accuracy
- ✓ Constraint 55-65% accuracy, A_activation > 2.0
- ✓ Naive clamp <5% accuracy (catastrophic)
- ✓ Mean-preserving ≥90% of constraint (preserves performance)

### Experiment B
- ✓ Critical ablation: Compensation > 1.3× in layers 10-12
- ✓ Random ablation: Compensation ≈ 1.0× (no effect)

### Experiment C
- ✓ High WD: Phase transition visible, ρ(acc, A_param) < -0.3
- ✓ No WD: A_param stays high, no sharp transition

### Experiment D
- ✓ Suppressor variance > 2.0× clean variance
- ✓ Bootstrap CI excludes 1.0 with 95% confidence

---

## Statistical Analysis

### Correlation Analysis (Exp C)

```python
from scipy.stats import spearmanr
import numpy as np

# Load time series
steps, acc, A_param = extract_time_series(log_file)

# Compute time derivatives
acc_velocity = np.gradient(acc)
A_param_velocity = np.gradient(A_param)

# Test anti-correlation during transition
transition_idx = (acc > 0.15) & (acc < 0.85)
rho, p_value = spearmanr(
    acc_velocity[transition_idx],
    A_param_velocity[transition_idx]
)

print(f"Correlation: ρ = {rho:.3f}, p = {p_value:.2e}")
# Expected: ρ < -0.3, p < 0.01
```

### Effect Size (Exp A)

```python
from lib.logging_utils import MetricAggregator

aggregator = MetricAggregator()

# Add runs
aggregator.add_run("constraint", 0.62)
aggregator.add_run("naive_clamp", 0.02)

# Compute Cohen's d
d = aggregator.compute_cohens_d("constraint", "naive_clamp")
print(f"Cohen's d: {d:.2f}")
# Expected: d > 4.0 (very large effect)
```

---

## Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
```bash
# Reduce batch size
python clean_audit/experiments/exp_a_foundation.py --batch_size 32

# Use CPU
python clean_audit/experiments/exp_a_foundation.py --device cpu
```

**Issue: TransformerLens not found**
```bash
pip install transformer-lens --upgrade
```

**Issue: Slow training (Exp C)**
```bash
# Use quick test mode
python clean_audit/experiments/exp_c_grokking.py --quick_test

# Reduce steps
python clean_audit/experiments/exp_c_grokking.py --n_steps 10000
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{allostatic_audit_2024,
  title={The Conservation of Separability: Allostatic Load in Transformer Networks},
  author={Research Team},
  year={2024},
  url={https://github.com/your-org/allo-audit}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Contact

For questions, issues, or contributions:
- GitHub Issues: [Link to issues page]
- Email: [Contact email]
- Research Group: [Lab website]

---

## Acknowledgments

This research builds on:
- **TransformerLens** by Neel Nanda (Anthropic)
- **ARENA** mechanistic interpretability curriculum
- **TinyStories** dataset by Eldan & Li (Microsoft Research)
- The broader mechanistic interpretability community

---

## Quick Start Guide

### First-Time Setup (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quick test of Experiment A
python clean_audit/experiments/exp_a_foundation.py --quick_test

# 3. Check output
ls clean_audit/data/

# 4. If successful, run full experiments
python clean_audit/experiments/exp_a_foundation.py
python clean_audit/experiments/exp_b_repair.py
python clean_audit/experiments/exp_c_grokking.py
python clean_audit/experiments/exp_d_superposition.py
```

### Full Experimental Run (6-8 hours on GPU)

```bash
# Run all experiments with seed 42
for exp in a b c d; do
    python clean_audit/experiments/exp_${exp}_*.py --seed 42
done

# Generate figures
python -c "from lib.plotting import create_all_figures; create_all_figures('clean_audit/data', 'clean_audit/figures')"

# Check success criteria
grep "SUCCESS" clean_audit/data/*.json
```

---

**Status:** Implementation Complete ✓
**Last Updated:** 2024-01-15
**Version:** 2.1
