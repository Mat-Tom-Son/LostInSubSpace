# Experiment D Implementation Summary

## File Location
`clean_audit/experiments/exp_d_superposition.py`

## Overview
Experiment D implements **functional suppressor identification** for the Allostatic Load research project. The core innovation is avoiding circular logic by defining suppressors via **ablation** (not by variance measurement), then measuring variance across different head types to validate the hypothesis.

## Key Components

### 1. Data Loading: `SimpleTokenDataset`
- Generates random token sequences for variance analysis
- Configurable vocabulary size, number of samples, and sequence length
- Returns input tokens and target (next token prediction)
- Minimum: >=1000 diverse examples (default: 2000)

### 2. Model Architecture: `SimpleTrans`
- Minimal multi-layer transformer for experimental clarity
- Configuration:
  - 2 transformer layers (focus on layer 0)
  - 4 attention heads per layer
  - d_model = 64 (configurable)
  - Standard transformer blocks with residual connections
  - LayerNorm + Multi-head attention + Feed-forward per block

### 3. Supporting Model Components
- **`MultiHeadAttention`**: Standard scaled dot-product attention
  - Stores attention patterns for analysis
  - Outputs [batch, seq, d_model]
- **`FeedForward`**: Position-wise MLP layer
  - 2-layer feed-forward with ReLU activation

### 4. Core Algorithm: `SuppressorAnalyzer`

#### 4.1 Functional Suppressor Identification (ANTI-CIRCULAR)
```
For each head in layer 0:
  1. Measure baseline downstream variance (var_baseline)
  2. Ablate the head (zero out its output)
  3. Measure ablated variance (var_ablated)
  4. If var_ablated > var_baseline * 1.5:
     -> Mark as SUPPRESSOR
```

**Why this avoids circular logic:**
- We do NOT use variance to define suppressors
- We use FUNCTIONAL ablation to identify which heads suppress variance
- The suppressor increases variance when removed (it was holding variance down)
- Only THEN do we measure variance to validate

#### 4.2 Key Methods
- **`measure_downstream_variance(batch, layer_idx)`**: Computes sigma-squared in residual stream after a layer
- **`ablate_head(batch, layer_idx, head_idx)`**: Zeros out a head's contribution, measures resulting variance
- **`identify_suppressors(dataloader, layer_idx, threshold=1.5)`**: Main ablation loop
  - Collects 20+ batches of measurements
  - Computes mean baseline and ablated variances per head
  - Identifies suppressors where ratio > threshold
  - Returns suppressor indices and detailed statistics

#### 4.3 Cross-validation: Structural Analysis
- **`compute_head_correlations(batch, layer_idx)`**: Correlation matrix of head outputs
- Success criterion: Suppressors show anti-correlation (correlation < -0.5)
- Validates that suppressors are functionally distinct

#### 4.4 Variance Measurement by Site
- **Suppressor heads** (layer 0, suppressor indices)
- **Clean heads (early)** (layer 0, non-suppressors)
- **Clean heads (late)** (layer 1, all heads)

### 5. Statistical Validation: Bootstrap Confidence Intervals

#### 5.1 `compute_bootstrap_ci(data, n_bootstrap=1000, ci=0.95)`
- Non-parametric estimation of mean and CI
- 1000 bootstrap resamples (default)
- Returns: (mean, lower_ci, upper_ci)

#### 5.2 Success Criteria
1. **Variance ratio**: Suppressor variance > 2.0x clean variance
2. **Bootstrap CI**: Does NOT include 1.0 (95% confidence)
3. **Anti-correlation**: Suppressors show correlation < -0.5

### 6. Main Experiment Flow: `main(args)`

```
Step 1: Setup (logging, reproducibility, device)
Step 2: Create data (>=1000 samples)
Step 3: Create and optionally pre-train model
Step 4: IDENTIFY SUPPRESSORS (via ablation)
        - Baseline variance measurement
        - Ablation for each head
        - Statistical thresholding
Step 5: MEASURE VARIANCE at three sites
        - Suppressor heads
        - Clean heads (early)
        - Clean heads (late)
Step 6: COMPUTE STATISTICS
        - Bootstrap CI for each site
        - Suppressor/clean ratio
        - Correlation analysis
Step 7: SAVE RESULTS
        - JSON output with all statistics
        - Summary to console
```

## Command-Line Arguments

```
# Data
--vocab_size        (default: 256)
--n_samples         (default: 2000, must be >=1000)
--seq_len           (default: 32)
--batch_size        (default: 32)

# Model
--d_model           (default: 64)
--n_heads           (default: 4)
--n_layers          (default: 2)
--d_ff              (default: 256)
--dropout           (default: 0.1)

# Experiment
--train_steps       (default: 200, pre-training iterations)
--suppressor_threshold  (default: 1.5, ratio for ablation)
--n_bootstrap       (default: 1000, bootstrap samples)

# System
--seed              (default: 42)
--use_cuda          (flag, use GPU if available)
--output_dir        (default: "clean_audit/data")
```

## Example Usage

```bash
# Basic run with defaults
python clean_audit/experiments/exp_d_superposition.py

# Custom parameters
python clean_audit/experiments/exp_d_superposition.py \
  --n_samples 2000 \
  --vocab_size 256 \
  --d_model 64 \
  --n_heads 4 \
  --n_layers 2 \
  --train_steps 300 \
  --suppressor_threshold 1.5 \
  --seed 42 \
  --use_cuda

# Minimal run (fast, fewer samples)
python clean_audit/experiments/exp_d_superposition.py \
  --n_samples 500 \
  --train_steps 50
```

## Output

Results are saved to `clean_audit/data/audit_log_exp_d_superposition_seed_*.json`

### Summary Statistics
```json
{
  "num_suppressors": "<int>",
  "suppressor_indices": "[<list of head indices>]",
  "suppressor_head_variance": {
    "mean": "<float>",
    "lower_ci": "<float>",
    "upper_ci": "<float>",
    "std": "<float>",
    "n_samples": "<int>"
  },
  "clean_heads_early_variance": {...},
  "clean_heads_late_variance": {...},
  "variance_ratio_suppressor_to_clean": "<float>",
  "suppressor_anti_correlation_count": "<int>"
}
```

## Library Dependencies

### Core Libraries (from `clean_audit/lib/`)
- **`metrics.py`**: `AllostasisAudit` class for variance computation
- **`logging_utils.py`**: `AuditLogger`, `setup_reproducibility` for experiment tracking

### External
- PyTorch: torch, torch.nn, torch.nn.functional
- NumPy: numpy for statistics
- Standard: argparse, tqdm, json, pathlib, typing

## Design Principles

### 1. **No Circular Logic**
- Suppressors defined by ABLATION effect, not by variance
- Variance measurement comes AFTER functional identification
- Clear separation of definition from validation

### 2. **Functional Clarity**
- Each component has a single, well-defined responsibility
- Ablation hook clearly separates head contributions
- Residual caching at layer boundaries for precise measurements

### 3. **Statistical Rigor**
- Multiple measurement sites (suppressor, clean early, clean late)
- Bootstrap CI for robustness
- Cross-validation via correlation analysis

### 4. **Reproducibility**
- Seed control
- Deterministic ablation procedure
- Complete logging to JSON

## Success Criteria (from Research Directive)

The experiment succeeds if:

1. **Suppressor Identification**: Heads identified with var_ablated > var_baseline * 1.5
2. **Variance Separation**: Suppressor variance >= 2.0x clean head variance
3. **Statistical Significance**: Bootstrap CI excludes 1.0 at 95% confidence
4. **Structural Validation**: Suppressors show anti-correlation (correlation < -0.5)

## Code Statistics
- **Total Lines**: 891
- **Classes**: 8 (SimpleTrans, MultiHeadAttention, FeedForward, SimpleTokenDataset, SuppressorAnalyzer, etc.)
- **Methods**: 30+ core methods
- **Comments**: Comprehensive docstrings for all classes and methods

## Notes

1. The model is intentionally simple for experimental clarity
2. Ablation is implemented via PyTorch hooks for clean separation
3. Variance measurements use the existing `AllostasisAudit.compute_variance()` method
4. Bootstrap CI provides non-parametric confidence bounds
5. All measurements are cached to avoid redundant forward passes
