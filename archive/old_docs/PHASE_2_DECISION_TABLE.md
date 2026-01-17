# PHASE 2.2 DECISION TABLE
**Created: 2026-01-12 before results**
**Status: LOCKED - Do not modify during analysis**

## Hard Thresholds (Defined Before Results)

### Amplitude Spike Criteria
A_activation is considered "elevated" if:
- **≥1.5x baseline** (Control A_act ≈ 11.5, so spike = ≥17.25)
- **Sustained for ≥3 consecutive logged epochs**
- **First derivative > 0.5 per epoch for ≥2 epochs**

### Accuracy Collapse Criteria
Performance is considered "collapsed" if:
- **Final accuracy < 20%** (vs ~88% constraint baseline from Phase 1.1)
- **Degradation > 50% from no-noise baseline**

### Clamp Divergence Criteria
Naive Clamp is diverging from Constraint if:
- **Accuracy gap ≥10 percentage points at final epoch**
- **Gap widening (not stable) over last 20 epochs**

## Decision Matrix

| Observation Pattern | Interpretation | Regime Classification |
|---------------------|----------------|----------------------|
| **1. A_act ↑ BEFORE acc ↓** | Amplitude compensation attempted, failed to prevent collapse | **Regime III (Partial)** - Compensation exists but insufficient |
| **2. Acc ↓ with NO A_act ↑** | Geometry failed, amplitude never recruited | **Outcome B (Geometry Wins)** - No compensatory pathway |
| **3. Naive Clamp < Constraint as noise ↑** | Amplitude freedom causally necessary | **Regime III (Evidence)** - Blocking amplitude hurts |
| **4. Naive Clamp ≈ Constraint across all noise** | Amplitude irrelevant, geometry exhaustion is cause | **Outcome B (Geometry Only)** - Amplitude not in the causal chain |
| **5. A_act ↑ AND acc stable** | Amplitude successfully compensating | **Regime III (Full)** - Conservation of separability via amplitude |

## Per-Noise-Level Predictions

### Noise = 0.05 (Low)
- **Expected**: Geometry survives, minimal stress
- **A_activation**: Flat or slight increase (< 1.2x)
- **Accuracy**: 80-90% (constraint), minimal degradation
- **Clamp vs Constraint**: No divergence (<5pp gap)

### Noise = 0.10 (Medium)
- **Expected**: Geometry stressed, may trigger compensation
- **A_activation**: Moderate increase (1.2-1.5x) if compensation exists
- **Accuracy**: 60-80% (constraint), noticeable degradation
- **Clamp vs Constraint**: Possible divergence (5-10pp) if amplitude recruited

### Noise = 0.15 (High)
- **Expected**: Geometry failing, amplitude MUST engage or collapse
- **A_activation**: Significant spike (≥1.5x) if Regime III exists, else flat
- **Accuracy**: 30-60% (constraint), severe degradation
- **Clamp vs Constraint**: Clear divergence (≥10pp) if amplitude causal
- **CRITICAL CASE**: This is where we expect to see Regime III if it exists

### Noise = 0.20 (Very High)
- **Expected**: Learning collapse regardless of mechanism
- **A_activation**: Either maxed out or flat (both indicate failure)
- **Accuracy**: <30% across all conditions
- **Clamp vs Constraint**: May converge to floor (both broken)

## Key Temporal Orderings to Check

### Ordering A: Amplitude Precedes Collapse
```
Epochs 0-30: Geometry working, A_act flat
Epochs 30-60: Geometry stressed, A_act rises
Epochs 60-90: A_act plateaus, accuracy dropping
Epochs 90-100: Collapse despite high A_act
```
**Interpretation**: Amplitude was recruited but insufficient → Partial Regime III

### Ordering B: Collapse Without Amplitude
```
Epochs 0-50: Geometry working, A_act flat
Epochs 50-70: Accuracy dropping, A_act still flat
Epochs 70-100: Collapsed, A_act never moved
```
**Interpretation**: No amplitude pathway → Outcome B

### Ordering C: Amplitude Rescues
```
Epochs 0-40: Geometry stressed, accuracy dropping
Epochs 40-60: A_act spikes sharply
Epochs 60-100: Accuracy stabilizes at reduced level, A_act sustained
```
**Interpretation**: Successful compensation → Full Regime III

## Critical Comparisons (Post-Run)

### Comparison 1: A_activation Trajectory
Plot A_act vs epoch for all 4 noise levels, Constraint condition only.
- **X-axis**: Epoch (0-100)
- **Y-axis**: A_activation
- **Lines**: One per noise level (0.00, 0.05, 0.10, 0.15, 0.20)
- **Baseline**: Horizontal line at Phase 1.1 control value (11.47)
- **Spike threshold**: Horizontal line at 1.5x baseline (17.25)

**Question**: Do higher noise levels show monotonic increase in A_act?

### Comparison 2: Constraint vs Naive Clamp Accuracy
Plot accuracy difference (Constraint - Naive Clamp) vs noise level.
- **X-axis**: Noise level (0.00, 0.05, 0.10, 0.15, 0.20)
- **Y-axis**: Accuracy gap (percentage points)
- **Positive gap**: Constraint outperforms (amplitude being used)
- **Zero gap**: No difference (amplitude not causal)

**Question**: Does the gap widen as noise increases?

### Comparison 3: FFN/Attn Variance Ratio
Plot FFN/Attn ratio vs epoch for noise 0.15 (critical case).
- **X-axis**: Epoch
- **Y-axis**: FFN/Attn variance ratio
- **Baseline**: Phase 1.1 value (≈3.7 for constraint)

**Question**: Does geometric bypass collapse as noise increases?

### Comparison 4: SNR vs Performance
Scatter plot: SNR (dB) vs final accuracy across all conditions/noise levels.
- **X-axis**: SNR in dB
- **Y-axis**: Final validation accuracy
- **Color**: Condition (control/constraint/clamp)

**Question**: Is there a sharp SNR threshold where performance falls off?

## Three Pre-Written Outcome Interpretations

### OUTCOME A: Regime III Observed
**Evidence Required:**
- A_activation ≥1.5x baseline at noise 0.15 or 0.20
- Naive Clamp < Constraint by ≥10pp at noise 0.15+
- Temporal ordering: A_act ↑ precedes or coincides with acc ↓

**Interpretation:**
> Phase 2.2 demonstrates partial evidence for Regime III. Under high noise (0.15+), amplitude scaling was recruited as a compensatory mechanism. The divergence between Constraint and Naive Clamp confirms amplitude freedom is causally necessary. However, compensation was insufficient to prevent collapse, suggesting the system reached the limits of the amplitude pathway before geometric exhaustion was complete.

**Next Step:** Phase 3 - Amplitude causality confirmed, test clamp severity sweep

---

### OUTCOME B: Geometry Always Wins
**Evidence Required:**
- A_activation remains < 1.5x baseline across all noise levels
- Naive Clamp ≈ Constraint (gap < 5pp) across all noise
- Accuracy collapses without amplitude engagement

**Interpretation:**
> Phase 2.2 falsifies the amplitude compensation hypothesis in this architecture. Across all noise levels, the system never recruited amplitude scaling. Performance degraded monotonically with noise, but A_activation remained flat. The equivalence between Constraint and Naive Clamp conditions proves amplitude is not in the causal chain. Collapse occurs when geometric routing fails, not when amplitude is blocked.

**Conclusion:** Overparameterized 1-layer transformers route around interference indefinitely via geometry. Amplitude is not a usable control variable in this regime.

**Next Step:** Phase 4 - Remove FFN entirely to test if attention-only forces amplitude

---

### OUTCOME C: Narrow Window
**Evidence Required:**
- A_activation rises modestly (1.2-1.4x) at noise 0.10-0.15
- Some clamp divergence (5-10pp) at noise 0.10-0.15
- Both signals disappear at noise 0.20 (gradient death)

**Interpretation:**
> Phase 2.2 reveals a narrow operational window where amplitude compensation is attempted but unreliable. At moderate noise (0.10-0.15), we observe mild A_activation increases and slight Naive Clamp underperformance, suggesting the system tries to recruit amplitude. However, this window is bracketed: too little noise and geometry suffices; too much noise and gradients collapse before amplitude can stabilize.

**Conclusion:** Compensatory amplitude scaling may exist but is inaccessible via SGD in noisy environments. The regime requires conditions narrower than theorized.

**Next Step:** Phase 4 - Test whether FFN removal widens the window

---

## Validation Checklist (Run After Results)

- [ ] Extract final accuracy for all 16 conditions (4 noise × 4 conditions)
- [ ] Extract A_activation trajectories from logs
- [ ] Calculate accuracy gaps (Constraint - Naive Clamp) per noise level
- [ ] Identify maximum A_activation per noise level
- [ ] Check temporal ordering: does A_act spike before accuracy drops?
- [ ] Plot Comparison 1 (A_act vs noise)
- [ ] Plot Comparison 2 (Clamp gap vs noise)
- [ ] Plot Comparison 3 (FFN/Attn ratio for noise 0.15)
- [ ] Plot Comparison 4 (SNR vs accuracy)
- [ ] Match observed pattern to Decision Matrix
- [ ] Select appropriate pre-written interpretation
- [ ] Classify as Outcome A, B, or C

---

**DO NOT MODIFY THIS TABLE AFTER VIEWING RESULTS**

Any post-hoc adjustments to thresholds or interpretations invalidate the experiment.
