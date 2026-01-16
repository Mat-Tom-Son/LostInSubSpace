# THE STRANGULATION TEST
**Pre-Registered Interpretation Framework**
**Date: 2026-01-12**
**Status: LOCKED - Do not modify after viewing results**

---

## Hypothesis

**"When geometric capacity is exhausted, the system will recruit amplitude scaling as a compensatory mechanism."**

## The Test

Reduce model dimension to **d=16** (from 128):
- **d_model = 16** (geometric capacity severely constrained)
- **d_ff = 16** (no expansion, 1:1 ratio)
- **vocab_size = 4096** (n/d = 256x, extreme superposition pressure)
- **noise_scale = 0.2** (maximum stress)
- **3/4 attention heads frozen** (constraint condition)
- **50 epochs** (reduced from 100 to save time)

## Why This Works

**Geometric constraint**: With d=16, the model has only 16 dimensions to represent:
- 4096 vocabulary items
- 128 sequence positions
- Multiple attention heads

**Physical intuition**: The model is in a "narrow tunnel" where geometric routing is barely possible. To push signal above the noise floor, it MUST increase amplitude.

**The Pre-LayerNorm forcing function**: After LayerNorm normalizes variance, the only way to increase SNR is to shift the mean (amplitude).

## Risks

**Risk 1: Gradient Collapse**
If d=16 is TOO constrained, gradients may vanish before learning converges. The model might not learn ANYTHING, let alone the energetic solution.

**Risk 2: Task Triviality**
If the interleaved sequence task is fundamentally easy, the model might solve it at d=16 without needing amplitude.

---

## Pre-Registered Outcomes

### OUTCOME A: The "Scream" ✓ SUCCESS

**Observations:**
- Final accuracy > 50% (constraint condition)
- A_activation spikes > 15.0 (vs baseline ~11.3)
- Naive Clamp < Constraint by ≥10pp (blocking amplitude hurts)

**Interpretation:**
> **Proof of Regime III Established.**
> When geometric capacity was exhausted (d=16), the system recruited amplitude scaling as the compensatory mechanism. The A_activation spike demonstrates that the model learned to "scream" to overcome the noise floor. The divergence between Constraint and Naive Clamp proves amplitude freedom is causally necessary.

**Physical meaning:**
The "Scream" is the physical manifestation of SIR maximization under geometric exhaustion. Separability (Ψ) is conserved via amplitude (A) when geometry (G) is unavailable.

**Next steps:**
- Document the phase transition threshold (what d triggers the scream?)
- Test amplitude-geometry trade-off curve
- Publish as primary result

**Classification:** Regime III Observed (Full)

---

### OUTCOME B: The "Silence" ⚠️ INCONCLUSIVE

**Observations:**
- Final accuracy < 10% (learning collapse)
- A_activation remains < 12.0 (no spike)
- All conditions perform poorly

**Interpretation:**
> **Geometric capacity is a prerequisite for learning.**
> The test did NOT falsify the amplitude compensation hypothesis. Instead, it revealed that d=16 is below the minimum capacity required for gradient descent to find ANY solution. The model collapsed before it could discover the energetic pathway.

**Analysis:**
This is the "gradient death before amplitude" scenario from the decision table. The system never reached the regime where amplitude could engage because learning itself failed.

**Why this doesn't disprove the theory:**
- Amplitude compensation requires a functioning gradient signal
- If d is TOO small, the model can't learn to represent the task at all
- The energetic solution exists but is unreachable via SGD from random init

**Next steps:**
- **Annealing protocol required**: Train at d=32 → project to d=16
- This gives gradients space to find a solution, THEN tests if amplitude can sustain it under compression
- Alternative: Test intermediate dimensions (d=24, d=32) to find the threshold

**Classification:** Outcome C (Narrow Window) - Compensation pathway may exist but is inaccessible

---

### OUTCOME C: The "Miracle" ⚠️ TASK FALSIFICATION

**Observations:**
- Final accuracy > 80% (constraint condition)
- A_activation remains < 12.0 (no spike)
- Constraint ≈ Naive Clamp (amplitude not causal)

**Interpretation:**
> **The interleaved sequence task is geometrically trivial.**
> If the model solves the task at d=16 without amplitude scaling, it means the task requires minimal geometric bandwidth. The "Narrow Road" was actually a highway. Geometric routing alone was sufficient even in 16 dimensions.

**Implications:**
- Cannot test "geometric exhaustion" on a task that costs nothing to solve
- The stress conditions (frozen heads, noise) were not sufficient to force true exhaustion
- Need a harder task: Modular arithmetic, multi-hop reasoning, or adversarial sequences

**Next steps:**
- Switch to modular addition task (requires true geometric structure)
- Or: Reduce d further (d=8, d=4) to find actual bottleneck
- Or: Add adversarial noise that specifically targets geometric structure

**Classification:** Outcome B (Geometry Wins) - But for the wrong reason (task too easy)

---

## Critical Thresholds (Pre-Defined)

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| **A_activation spike** | ≥ 15.0 | Amplitude engaged (vs baseline 11.3) |
| **Accuracy collapse** | < 10% | Learning failed |
| **Accuracy success** | > 50% | Model found solution |
| **Clamp divergence** | ≥ 10pp | Amplitude causally necessary |

## Validation Checklist

After results arrive:

- [ ] Extract final accuracy for all 4 conditions
- [ ] Extract A_activation trajectory (all epochs)
- [ ] Identify maximum A_activation per condition
- [ ] Calculate Constraint - Naive Clamp gap
- [ ] Check if A_activation exceeds 15.0 threshold
- [ ] Match observed pattern to one of three outcomes
- [ ] DO NOT modify interpretation post-hoc

---

## If Outcome B (Silence), Next Protocol

**Annealing Test:**
1. Train model at d=32 for 50 epochs (gives gradients room)
2. Save checkpoint
3. Project weights to d=16 (discard half of dimensions)
4. Continue training at d=16 for 50 epochs
5. Measure: Does A_activation spike during d=16 phase?

**Logic**: If amplitude exists, this should reveal it. The model learns a geometric solution at d=32, then is forced to compress. If Ψ = G + A holds, amplitude should rise to compensate for lost geometry.

---

**DO NOT MODIFY THIS DOCUMENT AFTER VIEWING RESULTS**

Any changes to thresholds or interpretations post-hoc invalidate the experiment.
