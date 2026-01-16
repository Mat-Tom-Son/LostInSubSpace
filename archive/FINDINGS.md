# Stroke Test & Sedation Experiment Findings

## TL;DR

**Q: Is amplitude scaling (A) a reflex or learned?**
**A: Learned, but NECESSARY.**

The model doesn't reflexively spike amplitude when injured. However, high-precision tasks naturally train models to maintain high amplitude as a defensive buffer against noise.

---

## Key Discovery: The Prophylactic Amplitude Law

| Task Type | A_activation | Reason |
|-----------|--------------|--------|
| Interleaved (Low Precision) | 8.13 | Geometric slack allows sloppiness |
| Modular Arithmetic (High Precision) | **11.64** | Zero slack requires exactness |

**40% more energy for high-precision tasks.**

---

## Definitive Sedation Test

Forced the high-amplitude modular model down to A=8.0, then added noise:

| Condition | Train Acc | Val Acc |
|-----------|-----------|---------|
| Healthy (A=11.6) | 100% | 100% |
| Sedated (A=8.0) + Noise | **89.1%** | **91.3%** |

**Proof:** High amplitude is not arbitrary — it's a structural requirement.

---

## The Precision Dial (Amplitude vs Density)

Higher precision demand (modulus density) drives amplitude up monotonically:

| Modulus | A_activation |
|---------|--------------|
| Interleaved (Slack) | 8.13 |
| p=7 | 11.41 |
| p=227 | 11.74 |

---

## The Mechanism: Geometry vs Amplitude

**Question:** Does training under noise force the model to learn even higher amplitude?

**Answer: NO.**
In a crowded regime (`d=32`, `p=113`), training with noise made the model robust (99.9% acc vs 93.4%), but:
- **Amplitude stayed flat** (4.33 vs 4.35).
- **Geometry changed completely** (Cosine Similarity = -0.006).

**Conclusion:** When forced to adapt, the model prefers to **rewire its geometry (G)** rather than spend more energy on amplitude (A). Amplitude is a margin buffer for constrained geometries, not the primary plasticity mechanism.

---

## The "Smoking Gun" Verification

How did the Hardened model achieve 99.9% robustness without increasing amplitude?

1.  **Geometric Rotation:**
    - Attention Similarity: **0.94** (Mechanics unchanged)
    - Residual Similarity: **-0.006** (Representations Rotated)
    - *The model rotated the semantic space to avoid noise.*

2.  **Sedation Check:**
    - Clamping Hardened model to A=3.0 under noise causes **Death (53% Acc)**.
    - *Amplitude is still the necessary fuel for the engine, even with a better steering wheel.*

---

## Run Your Own Tests

```bash
# Sedation test (the definitive experiment)
python clean_audit/experiments/exp_a_foundation.py \
    --dataset modular \
    --load_model healthy_victim_modular.pt \
    --vocab_size 128 --d_model 128 \
    --clamp_type naive --target_norm 8.0 \
    --post_clamp_noise_scale 2.0 \
    --n_epochs 1 --high_freq_log
```

See [STROKE_TEST.md](STROKE_TEST.md) for the full experimental protocol.
