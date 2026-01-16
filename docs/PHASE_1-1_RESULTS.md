# Phase 1.1 Results  
## FFN Crippling Experiment  
**Date:** 2026-01-12  

---

## Objective

Phase 1.1 was designed to test whether **amplitude scaling** is a *causally necessary* mechanism for compensation when geometric capacity is constrained.

Specifically, we asked:

> If we reduce the Feed-Forward Network’s geometric capacity, does the system escalate amplitude, or does it find another geometric route?

This phase was not expected to induce collapse. Its purpose was to identify **where compensation actually happens** when attention is constrained.

---

## Experimental Setup

### Architectural Intervention

- FFN expansion reduced from **4× to 1×**  
  - `d_ff = 512 → 128`  
- Attention unchanged  
- Same model, dataset, optimizer, and training schedule  
- Same constraint regime  
  - 3 of 4 attention heads frozen  
- Same noise level  
  - Noise = 0.0  
- Added variance tracking  
  - Attention output variance  
  - FFN output variance  

No other variables were modified.

---

## Hypothesis Under Test

If **amplitude scaling** is the primary compensatory mechanism, then reducing FFN capacity should force the system to escalate energy.

Predictions:

1. **Constraint condition**
   - Significant increase in `A_activation` (≥6× baseline)
2. **Naive Clamp**
   - Catastrophic failure (<5% accuracy)
3. **FFN reduction**
   - Forces attention to carry the load

---

## Results

### Quantitative Summary

| Condition | Accuracy | A_activation | FFN / Attn Variance | Interpretation |
|--------|----------|--------------|---------------------|----------------|
| Control | 99.17% | 11.47 | 0.00 | Healthy baseline, attention-dominated |
| Constraint | 88.23% | 11.35 ↓ | 3.73 ↑ | Geometric bypass via FFN |
| Naive Clamp | 84.81% | 11.37 | 3.09 | No collapse, mild degradation |
| Mean-Preserving Clamp | 84.08% | 10.45 ↓ | 2.99 | Variance suppression slightly harms performance |

---

## Critical Observations

### 1. No Amplitude Spike

- `A_activation` **did not increase** under constraint  
- It slightly **decreased** relative to control  
- This directly contradicts the hypothesis that amplitude scaling is required for compensation

**Expected:** Large activation spike  
**Observed:** Flat or declining activation  

---

### 2. No Clamp-Induced Collapse

- Naive Clamp accuracy remained ~85%  
- Training slowed but did not fail  
- System remained functional throughout

If amplitude were causally necessary, this clamp should have been fatal. It wasn’t.

---

### 3. Geometric Bypass Confirmed

- FFN / Attention variance ratio increased from **0.00 → 3.73**
- FFN became the dominant contributor to separability
- Compensation occurred through **geometry**, not energy

The system rerouted instead of escalating.

---

### 4. Counterintuitive Accuracy Improvement

| FFN Expansion | Constraint Accuracy |
|--------------|--------------------|
| 4× (d_ff=512) | ~44% |
| 1× (d_ff=128) | 88% |

Reducing FFN capacity **improved** constrained performance.

This result is paradoxical if FFN is merely a capacity or memorization module.

---

## Interpretation

### Law One: Generalized

> **As long as a geometric path exists, the system will take it.**

Phase 1.1 shows that geometry is **system-level**, not attention-specific.

When attention is constrained:
- The system does **not** escalate amplitude
- The FFN assumes the role of geometric routing
- Separability (Ψ) is preserved via a different geometric substrate

---

### FFN Is Not Just Memorization

Previous assumption:
- FFN = lookup table or capacity reservoir

Revised understanding:
- FFN = **implicit geometric transformation engine**

Evidence:
- FFN variance increases selectively under constraint
- Compensation preserves separability without increasing amplitude
- Same Ψ, different G

---

### Why Did Smaller FFN Perform Better?

With 4× expansion:
- The system had excessive geometric freedom
- The remaining attention head + FFN could trivially bypass constraints

With 1× expansion:
- Geometric freedom was reduced
- Bypass still existed, but was tighter and more efficient
- Performance improved under constraint

This suggests that *too much geometric slack can be counterproductive*.

---

## Implications

### What Phase 1.1 Established

- FFN is the **active geometric bypass**
- Amplitude is **not engaged**
- Norm clamps do **not** cause collapse when geometry remains
- Constraining attention alone is insufficient to force amplitude scaling

### What Remains Open

We have not yet observed **Regime III** (true compensatory amplitude scaling).

So far:
> Geometry always wins.

But we have only closed **one** geometric factory.

---

## Phase 2.2: Noise Sweep (In Progress)

### Motivation

Phase 1.1 demonstrated that constraining attention and FFN capacity is insufficient to force amplitude engagement as long as a geometric bypass remains available.

Phase 2.2 is designed to **directly attack the remaining geometric pathway** by injecting noise at the FFN input, after attention and LayerNorm, where it cannot be normalized away.

This phase explicitly tests whether amplitude scaling emerges **only after total geometric capacity (∑G) is exhausted**.

---

### Experimental Status

All experiments are currently running in parallel:

| Task ID | Noise Scale | Expected Regime |
|------|-------------|-----------------|
| be27926 | 0.05 | Geometry survives |
| b278f90 | 0.10 | Geometry stressed |
| b54838a | 0.15 | Geometry fails, compensation required |
| b2784af | 0.20 | Learning collapse |

Each noise level runs all four conditions:
- Control  
- Constraint  
- Naive Clamp  
- Mean-Preserving Clamp  

Training duration: 100 epochs per run.

---

### Pre-Committed Analysis Criteria

The following decision framework is locked **prior to results**:

- **Amplitude spike threshold:**  
  `A_activation ≥ 1.5× baseline (≥17.25)`
- **Clamp divergence threshold:**  
  ≥10 percentage point accuracy gap
- **Temporal ordering enforced:**  
  Activation changes must precede accuracy collapse to count as compensation

Three outcome interpretations (A, B, C) have been pre-written.  
No post-hoc threshold adjustments are permitted.

---

### Critical Diagnostic Questions

1. Does `A_activation` spike as noise increases?
   - If yes → amplitude compensation exists
   - If no → geometry remains dominant
2. When does accuracy collapse?
   - After activation rises → amplitude attempted and failed
   - Before activation rises → amplitude never engaged
3. Do Constraint and Naive Clamp diverge with noise?
   - Divergence → amplitude becoming causally necessary
   - No divergence → geometry still routing around constraints

Live monitoring is limited to:
- Noise levels **0.15 and 0.20**
- **Constraint condition**
- `A_activation` trajectory only

---

## Status

**Phase 1.1 Classification:** Outcome B (Provisional)  
**Current Regime:** Geometry-dominant  
**Phase 1.1 Success Criteria:** **MET**

- ✓ Identified FFN as the active bypass
- ✓ Confirmed absence of amplitude escalation
- ✓ Maintained reasonable performance under constraint
- ✓ Localized where geometry is being supplied

Phase 2.2 is now determining whether **amplitude is a real regime** or a theoretical ghost.

The system is under pressure.  
We are waiting to see if it ever gets loud.
