# STROKE TEST - POSTMORTEM AND CORRECTIONS

**Date: 2026-01-13**
**Status: First attempt INVALID - Parameters too weak**

---

## What Happened: The False Negative

### The Execution Was Perfect
✅ High-frequency logging worked flawlessly (every 10 batch steps)
✅ Model loading/saving worked correctly
✅ Constraints applied as intended (3/4 heads frozen)
✅ Noise injection worked (SNR tracked)

### But The Experiment Was Invalid

**The "Victim" Wasn't Healthy:**
```
Expected:  ~95% accuracy (confident, geometric solution established)
Actual:    76.5% accuracy (already struggling)
Problem:   d=40 was too constrained from the start
```

**The "Stroke" Wasn't Violent:**
```
Expected:  95% → 40% accuracy (crisis mode)
Actual:    76.5% → 75.5% accuracy (-1% drop)
Problem:   Model barely noticed the constraints
```

**No Amplitude Spike Because No Crisis:**
```
A_activation throughout: 6.49 (completely flat)
Expected spike:         6.0 → 12.0+ (2x increase)
Conclusion:             No distress = No reflex engaged
```

---

## Root Cause Analysis

### Problem 1: d=40 Is Fundamentally Too Small

**The Math:**
- n/d = 4096/40 = **102.4x** superposition pressure
- This is EXTREME - the model can barely learn even without constraints
- Think of it like trying to store 100 files in 1 byte

**The Evidence:**
- Control (no constraints) only reached 76.5% after 20 epochs
- Should have been able to reach 95%+ easily
- Model was already "strangled" before we even started

**The Analogy:**
> You can't give someone a stroke if they're already unconscious.
> We tried to strangle someone who was already strangling.

### Problem 2: noise_scale=0.15 Was Too Gentle

**The Math:**
- Signal power: ~1.08 (from variance)
- Noise power: 0.15² = 0.0225
- SNR: ~17dB (signal is 50x louder than noise)

**The Evidence:**
- Accuracy dropped by only 1%
- A_activation didn't budge
- Model just... ignored the noise

**The Analogy:**
> We threw a marshmallow at the model and expected it to flinch.

### Problem 3: 20 Epochs Wasn't Enough for "Addiction"

The victim needs to be **confident and addicted** to its geometric solution, so that breaking it causes panic.

- 20 epochs: Model is still figuring things out
- 50+ epochs: Model has converged, committed to a strategy
- When you break a committed strategy → crisis → reflex

---

## The Corrected Protocol

### Fix 1: Use d=64 (Not d=40)

**The Math:**
- n/d = 4096/64 = **64x** superposition pressure
- Still significant, but manageable
- Proven range: d=128 reaches 88%, so d=64 should reach 95%+

**Expected Result:**
- Control baseline: **95-98% accuracy**
- A_activation: ~6-7 (healthy, no stress)
- Model is confident, geometric solution established

### Fix 2: Use noise_scale=0.3 (Not 0.15)

**The Math:**
- Noise power: 0.3² = 0.09 (4x stronger)
- SNR will drop from 17dB → ~11dB
- This is the difference between "annoying" and "blinding"

**Expected Result:**
- Accuracy should **crash** from 95% → 40%
- Model should go effectively blind
- This is the "stroke" - sudden, catastrophic loss of function

### Fix 3: Train Victim for 50 Epochs (Not 20)

**The Reason:**
- 50 epochs ensures full convergence
- Model becomes "addicted" to its geometric solution
- When you break something the model relies on → panic

**Expected Result:**
- Victim reaches ceiling performance (~95-98%)
- Geometric solution is well-established
- Breaking it causes immediate crisis

---

## Updated Expectations

### Step 1: Train Healthy Victim (d=64, 50 epochs)

```
Epoch 0:  Random init, ~0% accuracy
Epoch 10: Learning geometry, ~40% accuracy
Epoch 20: Geometry working, ~70% accuracy
Epoch 30: Converging, ~85% accuracy
Epoch 50: Fully converged, ~95% accuracy

Final: A_activation ~6-7, variance stable
```

### Step 2: Apply Violent Stroke (noise=0.3, frozen heads)

```
Step 0:    Loaded healthy weights, ~95% accuracy, A_act ~6.5
Step 10:   Constraints hit, accuracy crashes to ~50%, A_act ???
Step 50:   Crisis mode, accuracy ~40%, A_act ???
Step 100:  Recovery attempt, accuracy ???, A_act ???

PREDICTION: A_activation spikes from 6.5 → 13+ to restore SNR
```

---

## The Three Possible Outcomes

### Outcome A: The Scream (Hypothesis A Confirmed)

**Observations:**
- Step 0-50: Accuracy crashes from 95% → 40%
- Step 50-100: **A_activation spikes from ~6.5 → 13+** (2x increase)
- Step 100+: Accuracy partially recovers to ~50-60%

**Interpretation:**
> **Amplitude is a reflex.** When geometric routes were suddenly blocked, the model immediately increased signal amplitude to maintain SNR. This happened BEFORE gradients could slowly adapt. The "scream" is the physical manifestation of an autonomous compensatory mechanism.

**Next Step:** Document the reflex threshold and response curve.

---

### Outcome B: The Silence (Hypothesis B Confirmed)

**Observations:**
- Step 0-50: Accuracy crashes from 95% → 40%
- Step 50-200: A_activation stays flat (~6.5, no spike)
- Step 200+: Gradual recovery via SGD, A_act slowly rises

**Interpretation:**
> **Amplitude is purely learned, not reflexive.** The model has no built-in mechanism to immediately boost amplitude under acute stress. Any amplitude scaling we've seen in previous experiments was gradual adaptation via gradient descent, not an autonomous reflex.

**Next Step:** Accept that amplitude is not a control variable at the acute timescale. Focus on geometric mechanisms.

---

### Outcome C: The Death (Experiment Still Invalid)

**Observations:**
- Accuracy crashes and never recovers (stays <10%)
- A_activation may spike OR stay flat, doesn't matter
- Gradient collapse, no learning

**Interpretation:**
> **We hit the victim too hard.** noise_scale=0.3 may be TOO violent, causing gradient death before the model can engage any mechanism. Need to back off to noise_scale=0.2 and try again.

**Next Step:** Reduce violence slightly, retry.

---

## How to Run the Corrected Test

**Option 1: Batch File (Simplest)**
```batch
RUN_STROKE_TEST_CORRECTED.bat
```

**Option 2: Python Script**
```bash
python run_stroke_test_corrected.py
```

**Option 3: Manual Commands**
```bash
# Step 1: Train victim
python clean_audit/experiments/exp_a_foundation.py \
    --condition control \
    --d_model 64 \
    --vocab_size 4096 \
    --n_epochs 50 \
    --seed 42 \
    --save_model healthy_victim_d64.pt

# Step 2: Apply stroke
python clean_audit/experiments/exp_a_foundation.py \
    --condition constraint \
    --load_model healthy_victim_d64.pt \
    --d_model 64 \
    --vocab_size 4096 \
    --n_epochs 5 \
    --noise_scale 0.3 \
    --seed 42 \
    --high_freq_log
```

---

## Expected Runtime

- **Step 1** (50 epochs, d=64): ~10-15 minutes
- **Step 2** (5 epochs, high-freq logging): ~2-3 minutes
- **Total**: ~15-20 minutes

---

## Lessons Learned

1. **"Healthy" means >95%, not 76%**
   - The victim must be genuinely confident before you can test crisis response

2. **"Stroke" means catastrophic drop, not 1%**
   - Need to see 95% → 40% to create genuine crisis
   - Small perturbations don't trigger crisis mechanisms

3. **Test the test before trusting the test**
   - We correctly implemented high-freq logging
   - But we didn't validate that the experimental conditions were strong enough

4. **The execution was perfect, the design was wrong**
   - This is why pre-registered protocols matter
   - We can clearly separate "tool works" from "parameters wrong"

---

## Status: READY FOR CORRECTED RUN

The code is solid. The protocol is now correct. Execute when ready.
