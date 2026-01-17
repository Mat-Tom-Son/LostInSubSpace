# Experiment D: Key Code Snippets

## 1. Suppressor Identification via Ablation (Core Algorithm)

The functional definition of suppressors via ablation:

```python
def identify_suppressors(self, dataloader, layer_idx=0, variance_ratio_threshold=1.5):
    """
    FUNCTIONAL DEFINITION (not circular):
    - For each head, measure baseline downstream variance
    - Ablate the head, measure ablated variance
    - If var_ablated > var_baseline * threshold -> suppressor
    """

    n_heads = self.model.n_heads
    baseline_variances = {i: [] for i in range(n_heads)}
    ablated_variances = {i: [] for i in range(n_heads)}

    # Collect measurements across batches
    for batch_idx, (input_ids, _) in enumerate(tqdm(dataloader)):
        if batch_idx >= 20:
            break

        # Baseline (no ablation)
        baseline_var = self.measure_downstream_variance(input_ids, layer_idx)
        for head_idx in range(n_heads):
            baseline_variances[head_idx].append(baseline_var)

        # Ablated (for each head)
        for head_idx in range(n_heads):
            ablated_var = self.ablate_head(input_ids, layer_idx, head_idx)
            ablated_variances[head_idx].append(ablated_var)

    # Identify suppressors: ratio > threshold
    suppressors = set()
    for head_idx in range(n_heads):
        base_var = np.mean(baseline_variances[head_idx])
        abl_var = np.mean(ablated_variances[head_idx])
        ratio = abl_var / base_var if base_var > 1e-10 else float('inf')

        is_suppressor = ratio > variance_ratio_threshold
        if is_suppressor:
            suppressors.add(head_idx)

    return suppressors, head_stats
```

## 2. Ablation Hook (Functional Implementation)

The PyTorch hook that zeros out a head's contribution:

```python
def ablate_head(self, batch, layer_idx=0, head_idx=0):
    """
    Measure downstream variance when a specific attention head is ablated.
    """
    def ablate_hook(module, input_args, output):
        # output: [batch, seq, d_model]
        # Zero out this head's contribution
        head_size = output.shape[-1] // self.model.n_heads
        start_idx = head_idx * head_size
        end_idx = (head_idx + 1) * head_size

        output_ablated = output.clone()
        output_ablated[:, :, start_idx:end_idx] = 0.0

        return output_ablated

    # Register hook
    layer = self.model.blocks[layer_idx]
    attn_module = layer[1]
    handle = attn_module.register_forward_hook(ablate_hook)

    try:
        with torch.no_grad():
            self.model.clear_cache()
            _ = self.model(batch.to(self.device))

            residual = self.model._cache[f'layer_{layer_idx}_residual']
            variance = self.auditor.compute_variance(residual)
    finally:
        handle.remove()

    return variance
```

## 3. Variance Measurement at Multiple Sites

```python
def measure_variance_by_site(model, dataloader, suppressors, device):
    """
    Measure variance at three sites:
    1. Suppressor heads (layer 0)
    2. Clean heads (early layer 0)
    3. Clean heads (late layer 1)
    """

    variances = {
        'suppressor_heads': [],
        'clean_heads_early': [],
        'clean_heads_late': []
    }

    for batch_idx, (input_ids, _) in enumerate(tqdm(dataloader, total=20)):
        if batch_idx >= 20:
            break

        # Suppressor heads (layer 0)
        for head_idx in suppressors:
            var = analyzer.measure_head_output_variance(
                input_ids, layer_idx=0, head_idx=head_idx
            )
            variances['suppressor_heads'].append(var)

        # Clean heads (early, layer 0)
        clean_early = set(range(model.n_heads)) - suppressors
        for head_idx in clean_early:
            var = analyzer.measure_head_output_variance(
                input_ids, layer_idx=0, head_idx=head_idx
            )
            variances['clean_heads_early'].append(var)

        # Clean heads (late, layer 1)
        for head_idx in range(model.n_heads):
            var = analyzer.measure_head_output_variance(
                input_ids, layer_idx=1, head_idx=head_idx
            )
            variances['clean_heads_late'].append(var)

    return variances
```

## 4. Bootstrap Confidence Intervals

```python
def compute_bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """
    Non-parametric confidence interval via bootstrap resampling.
    """
    data = np.array(data)
    mean = np.mean(data)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)
    alpha = 1.0 - ci
    lower_ci = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_ci = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return mean, lower_ci, upper_ci
```

## 5. Correlation Analysis (Cross-validation)

```python
def compute_head_correlations(self, batch, layer_idx=0):
    """
    Compute correlation matrix to validate suppressors show anti-correlation.

    Expected: rho < -0.5 for suppressor pairs
    """
    def capture_heads_hook(module, input_args, output):
        head_size = output.shape[-1] // n_heads
        for h in range(n_heads):
            start_idx = h * head_size
            end_idx = (h + 1) * head_size
            output_flat = output[:, :, start_idx:end_idx].reshape(-1)
            head_outputs[h] = output_flat.detach().cpu().numpy()
        return output

    layer = self.model.blocks[layer_idx]
    attn_module = layer[1]
    handle = attn_module.register_forward_hook(capture_heads_hook)

    try:
        with torch.no_grad():
            _ = self.model(batch.to(self.device))

            if len(head_outputs) == n_heads:
                output_matrix = np.array([head_outputs[i] for i in range(n_heads)])
                corr_matrix = np.corrcoef(output_matrix)
            else:
                corr_matrix = np.full((n_heads, n_heads), np.nan)
    finally:
        handle.remove()

    return corr_matrix
```

## 6. Main Experiment Orchestration

```python
def main(args):
    """Main Experiment D pipeline."""

    # Setup
    setup_reproducibility(args.seed)
    device = torch.device("cuda" if args.use_cuda else "cpu")
    logger = AuditLogger("exp_d_superposition", args.output_dir, args.seed)

    # Data
    dataset = SimpleTokenDataset(
        vocab_size=args.vocab_size,
        n_samples=args.n_samples,
        seq_len=args.seq_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Model
    model = SimpleTrans(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers
    ).to(device)

    # Pre-train (optional)
    if args.train_steps > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for step, (input_ids, target_ids) in enumerate(dataloader):
            if step >= args.train_steps:
                break
            logits = model(input_ids.to(device))
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size),
                                  target_ids.reshape(-1).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # IDENTIFY SUPPRESSORS (core algorithm)
    analyzer = SuppressorAnalyzer(model, device)
    suppressors, head_stats = analyzer.identify_suppressors(
        dataloader,
        layer_idx=0,
        variance_ratio_threshold=args.suppressor_threshold
    )

    # MEASURE VARIANCE at three sites
    variances = measure_variance_by_site(model, dataloader, suppressors, device)

    # COMPUTE STATISTICS
    statistics = {}
    for site, values in variances.items():
        mean, lower_ci, upper_ci = compute_bootstrap_ci(
            values, n_bootstrap=args.n_bootstrap, ci=0.95
        )
        statistics[site] = {
            'mean': float(mean),
            'lower_ci': float(lower_ci),
            'upper_ci': float(upper_ci),
            'n_samples': len(values),
            'std': float(np.std(values))
        }

    # Variance ratio
    if suppressors:
        ratio = np.mean(variances['suppressor_heads']) / np.mean(variances['clean_heads_early'])
        statistics['variance_ratio'] = ratio

    # Save results
    logger.log_metrics(0, {'num_suppressors': len(suppressors), 'stats': head_stats})
    logger.log_metrics(1, statistics)
    logger.save_log()

    print(f"Suppressors: {len(suppressors)}")
    print(f"Variance ratio: {ratio:.2f}x")
```

## Usage Examples

```bash
# Basic run with defaults
python clean_audit/experiments/exp_d_superposition.py

# Custom configuration
python clean_audit/experiments/exp_d_superposition.py \
  --n_samples 2000 \
  --d_model 64 \
  --n_heads 4 \
  --train_steps 200 \
  --suppressor_threshold 1.5 \
  --n_bootstrap 1000 \
  --seed 42 \
  --use_cuda

# Minimal run (fast testing)
python clean_audit/experiments/exp_d_superposition.py \
  --n_samples 500 \
  --train_steps 50 \
  --n_bootstrap 100
```

## Key Design Principles in Code

### No Circular Logic
```python
# WRONG: Define suppressor by variance, then measure variance
suppressors = [h for h in heads if variance(h) > threshold]

# CORRECT: Define suppressor by ablation, then measure variance
suppressors = [h for h in heads if ablate_effect(h) > threshold]
```

### Functional Clarity
Each method has single responsibility:
- `measure_downstream_variance()` - Compute sigma squared
- `ablate_head()` - Zero out head, measure effect
- `identify_suppressors()` - Collect measurements, identify
- `measure_variance_by_site()` - Measure at three sites
- `compute_bootstrap_ci()` - Statistical validation

### Reproducibility
```python
setup_reproducibility(args.seed)  # Seed all RNGs
logger = AuditLogger(...)  # Log to JSON
model.eval()  # Disable dropout
with torch.no_grad()  # No gradients
```
