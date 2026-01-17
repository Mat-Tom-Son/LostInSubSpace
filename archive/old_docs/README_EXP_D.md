# Experiment D: Polysemanticity and High-A Regime

## Quick Start

```bash
# Run with defaults (2000 samples, 4 heads, 2 layers)
python clean_audit/experiments/exp_d_superposition.py

# Run with custom parameters
python clean_audit/experiments/exp_d_superposition.py \
  --n_samples 2000 \
  --vocab_size 256 \
  --d_model 64 \
  --n_heads 4 \
  --train_steps 200 \
  --seed 42 \
  --use_cuda
```

## What This Experiment Does

### Goal
Link polysemanticity to the high-A (amplitude) regime by identifying **suppressor heads** that operate in high-variance conditions.

### Key Innovation: Functional Suppressor Identification
- Define suppressors via **ABLATION**, not by variance measurement
- Avoid circular logic: "suppress variance to define suppressors, then measure variance"
- A suppressor is identified when var_ablated > var_baseline * 1.5

### The Pipeline

1. **Identify suppressors via ablation**
   - For each attention head in layer 0:
     - Measure baseline downstream variance
     - Ablate the head (zero out its output)
     - Measure variance with ablation
     - If ablated > baseline * 1.5 → **SUPPRESSOR**

2. **Cross-validate structurally**
   - Compute correlation between head outputs
   - Suppressors should show anti-correlation (ρ < -0.5)
   - Validates functional distinctness

3. **Measure variance at three sites**
   - **Suppressor heads** (layer 0)
   - **Clean heads (early)** (layer 0, non-suppressors)
   - **Clean heads (late)** (layer 1)

4. **Statistical validation**
   - Bootstrap confidence intervals (1000 resamples)
   - Success: suppressor variance > 2.0x clean variance
   - Success: CI does not include 1.0 (95% confidence)

## Files

| File | Purpose |
|------|---------|
| `clean_audit/experiments/exp_d_superposition.py` | Main experiment script (891 lines) |
| `EXP_D_IMPLEMENTATION_SUMMARY.md` | Implementation details and architecture |
| `EXP_D_ARCHITECTURE.txt` | System architecture and data flow |
| `README_EXP_D.md` | This file |

## Architecture

### Core Classes

```
SimpleTokenDataset
  -> Random token sequences (>=1000 samples)

SimpleTrans
  -> 2-layer transformer
  -> 4 attention heads per layer
  -> d_model=64, d_ff=256

MultiHeadAttention
  -> Scaled dot-product attention
  -> Stores patterns for analysis

SuppressorAnalyzer
  -> measure_downstream_variance(batch, layer_idx)
  -> ablate_head(batch, layer_idx, head_idx)
  -> identify_suppressors(dataloader, threshold=1.5)
  -> compute_head_correlations(batch, layer_idx)
```

### Helper Functions

```
measure_variance_by_site(model, dataloader, suppressors)
  -> Returns {suppressor_heads, clean_heads_early, clean_heads_late}

compute_bootstrap_ci(data, n_bootstrap=1000, ci=0.95)
  -> Returns (mean, lower_ci, upper_ci)

main(args)
  -> Orchestrates entire experiment pipeline
```

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--n_samples` | 2000 | Must be >= 1000 |
| `--vocab_size` | 256 | Token vocabulary size |
| `--d_model` | 64 | Model dimension |
| `--n_heads` | 4 | Attention heads per layer |
| `--n_layers` | 2 | Number of transformer layers |
| `--train_steps` | 200 | Pre-training iterations |
| `--suppressor_threshold` | 1.5 | Ablation ratio threshold |
| `--n_bootstrap` | 1000 | Bootstrap samples for CI |
| `--seed` | 42 | Random seed |
| `--use_cuda` | False | Use GPU if available |

## Output

Results saved to: `clean_audit/data/audit_log_exp_d_superposition_seed_*.json`

### Example Output

```json
{
  "metrics": [
    {
      "step": 0,
      "num_suppressors": 2,
      "suppressor_indices": [0, 3],
      "suppressor_stats": {
        "0": {
          "baseline_variance": 0.234,
          "ablated_variance": 0.456,
          "variance_ratio": 1.95,
          "is_suppressor": true,
          "n_samples": 20
        }
      }
    },
    {
      "step": 1,
      "suppressor_heads_variance": {
        "mean": 0.425,
        "lower_ci": 0.412,
        "upper_ci": 0.438,
        "std": 0.032,
        "n_samples": 40
      },
      "clean_heads_early_variance": {
        "mean": 0.195,
        "lower_ci": 0.188,
        "upper_ci": 0.202,
        "std": 0.018,
        "n_samples": 40
      },
      "variance_ratio_suppressor_to_clean": 2.18
    }
  ]
}
```

## Success Criteria

From the research directive:

1. **Suppressor Identification** (via ablation)
   - var_ablated / var_baseline > 1.5 for identification

2. **Variance Separation**
   - Suppressor variance > 2.0x clean head variance

3. **Statistical Significance**
   - Bootstrap CI (95%) does NOT include 1.0

4. **Structural Validation**
   - Suppressors show anti-correlation (ρ < -0.5)

## How Ablation Works

The key innovation is the ablation hook:

```python
def ablate_hook(module, input_args, output):
    """Zero out a specific head's output."""
    head_size = output.shape[-1] // n_heads
    start_idx = head_idx * head_size
    end_idx = (head_idx + 1) * head_size

    output_ablated = output.clone()
    output_ablated[:, :, start_idx:end_idx] = 0.0

    return output_ablated
```

This:
- Cleanly separates each head's contribution
- Allows accurate measurement of suppressive effect
- Avoids circular logic (ablate -> measure, not measure -> define)

## Variance Measurement Sites

Why three sites?

1. **Suppressor heads** - Expected high variance (that's what they suppress)
2. **Clean heads (early)** - Baseline variance in layer 0
3. **Clean heads (late)** - Control from different layer

Comparison shows whether suppressors are genuinely different.

## Example Run Output

```
[EXP D] Creating dataset...
[EXP D] Creating model...
Model created with 33856 parameters

[EXP D] Pre-training model for 200 steps...
  Step 0: loss=5.5234
  Step 100: loss=3.2145
  Step 200: loss=2.1543

[EXP D] Identifying suppressors in layer 0...
Collecting variances: 100%|████████| 20/20
  Head 0: SUPPRESSOR (ratio=1.87)
  Head 1: clean (ratio=0.94)
  Head 2: clean (ratio=1.12)
  Head 3: SUPPRESSOR (ratio=1.92)

Identified 2 suppressors: {0, 3}

[EXP D] Measuring variance at multiple sites...
Measuring variances: 100%|████████| 20/20

[EXP D] Computing statistics...

suppressor_heads:
  Mean: 0.425234
  CI [95%]: [0.412451, 0.438123]
  Std: 0.032145
  N: 40

clean_heads_early:
  Mean: 0.195432
  CI [95%]: [0.188234, 0.202456]
  Std: 0.015234
  N: 40

[EXP D] Variance Ratio (Suppressor / Clean):
  Ratio: 2.18x
  SUCCESS: Ratio >= 2.0x (meets success criterion)

[EXP D] Computing head correlations...
  Suppressors 0-3: rho=-0.62 (anti-correlated)

======================================================================
EXPERIMENT D SUMMARY
======================================================================
Suppressors identified: 2
Suppressor indices: {0, 3}
Variance ratio (Supp/Clean): 2.18x
Bootstrap CIs computed: yes
Output directory: clean_audit/data
======================================================================
```

## Implementation Notes

### No Circular Logic
The design carefully avoids the circular definition problem:
- NOT: "Define suppressors by high variance, measure variance"
- CORRECT: "Define suppressors by ablation effect, measure variance to validate"

### Functional Clarity
Each component has clear responsibility:
- `SimpleTrans`: Model architecture
- `MultiHeadAttention`: Attention computation
- `SuppressorAnalyzer`: Ablation and identification
- `measure_variance_by_site`: Multi-site measurement
- `compute_bootstrap_ci`: Statistical analysis

### Reproducibility
- Deterministic random seed control
- All measurements logged to JSON
- Complete experiment metadata stored

## Dependencies

Core:
- PyTorch (torch, torch.nn, torch.nn.functional)
- NumPy (numpy)
- tqdm (progress bars)

From `clean_audit/lib/`:
- `metrics.py` (AllostasisAudit for variance)
- `logging_utils.py` (AuditLogger, setup_reproducibility)

## References

This experiment implements the research directive for Experiment D:
- Goal: Link polysemanticity to high-A regime
- Method: Functional suppressor identification via ablation
- Resource: Small transformer with >=1000 diverse tokens
- Success: Variance ratio >= 2.0x with CI excluding 1.0

## License

Same as main project.
