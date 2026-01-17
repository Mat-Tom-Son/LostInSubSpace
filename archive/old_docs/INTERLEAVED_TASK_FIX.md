# The Interleaved Task Fix - Forcing True Interference

## Status: ✅ IMPLEMENTED - Ready to Test

---

## The Problem We Solved

### Failure Mode: L < d (No Geometric Crowding)

**Previous runs:**
```
d_model=128, seq_len=32: L/d = 0.25  ← No crowding
d_model=64,  seq_len=32: L/d = 0.50  ← Still no crowding
d_model=40,  seq_len=32: L/d = 0.80  ← Close, but not enough
```

**Result:** Even with n >> d (vocab >> d_model), the model could separate all 32 context positions in 40+ dimensions without interference. Single head solves task geometrically → no amplitude compensation needed.

### The Mathematical Issue

```
Old Task: Simple repeating patterns
- seq_len = 32 token positions
- d_model = 40 dimensions
- 32 vectors in 40D space
- Perfect geometric separation possible
- Single head: "I can see all tokens clearly"
- No need to amplify signal
```

---

## The Solution: Dual Interference

### 1. Geometric Crowding (L > d)

```
New: seq_len = 128 (hardcoded in config)
     d_model = 40 (configurable)
     L/d = 3.2

128 token positions in 40 dimensions → vectors MUST overlap
```

### 2. Semantic Interference (Interleaved Streams)

```python
class InterleavedSequenceDataset:
    """
    Input:  [A1, B1, A2, B2, A3, B3, ...]
    Target: [B1, A2, B2, A3, B3, A4, ...]

    Challenge:
    - Two independent sequences (A and B) share the residual stream
    - To predict A3, must attend back to A2 (distance 2)
    - But B2 is in between (distance 1, "louder" signal)
    - With 1 head, must filter B-stream noise from A-stream signal
    - Requires amplitude scaling to boost relevant signal
    ```

**Why This Forces Compensation:**

1. **Geometric:** 128 positions can't all be orthogonal in 40D
2. **Semantic:** Two streams interfere in shared representation space
3. **Attention:** Single head must filter one stream from another
4. **Without amplitude:** Signals drown in cross-stream noise
5. **With amplitude:** Scale up relevant stream to overcome interference

---

## What Changed

### 1. New Dataset Class (Lines 208-254)

```python
class InterleavedSequenceDataset(Dataset):
    """Forces both geometric and semantic interference."""

    def __init__(self, n_samples=10000, seq_len=128, vocab_size=4096):
        # seq_len=128 by default (L > d for most configs)
        ...

    def __getitem__(self, idx):
        # Generate two independent streams
        seq_a = torch.randint(0, vocab_size, (half_len,))
        seq_b = torch.randint(0, vocab_size, (half_len,))

        # Interleave: A1, B1, A2, B2, ...
        interleaved[0::2] = seq_a
        interleaved[1::2] = seq_b
        ...
```

### 2. Updated Config (Line 556)

```python
config = {
    ...
    'seq_len': 128,  # Was 32, now 128 for L > d
    ...
}
```

### 3. Enhanced Validation (Lines 564-591)

```python
n_over_d = vocab_size / d_model
L_over_d = seq_len / d_model

print(f"  n/d ratio: {n_over_d:.1f}")
print(f"  L/d ratio: {L_over_d:.2f}")

if L_over_d < 1.5:
    print("  WARNING: No geometric crowding!")
else:
    print("  GOOD: Geometric crowding + semantic interference")
```

### 4. Dataset Switch (Lines 427, 432)

```python
# OLD: ToySequenceDataset (simple patterns)
# NEW: InterleavedSequenceDataset (forced interference)
train_dataset = InterleavedSequenceDataset(...)
val_dataset = InterleavedSequenceDataset(...)
```

---

## How to Run

### The Definitive Test

```bash
cd clean_audit
python experiments/exp_a_foundation.py --d_model 40 --vocab_size 4096 --n_epochs 50 --seed 42
```

**Expected Configuration:**
```
[CONFIG VALIDATION]
  n/d ratio: 102.4  ← Superposition forced
  L/d ratio: 3.20   ← Geometric crowding
  OPTIMAL: Both superposition AND crowding present
  Constraint should trigger allostatic load
```

**Expected Results:**

| Condition | Accuracy | A_act | Status |
|-----------|----------|-------|--------|
| Control | 65-80% | ~6-7 | Learns both streams |
| Constraint | 25-40% | **15-25** | Struggles, amplifies |
| Naive Clamp | <5% | ~6-7 | Collapse (A blocked) |
| Mean-Pres | 20-35% | **15-25** | Preserved (A allowed) |

**Key Predictions:**
- Constraint A_act should be **2-4× baseline**
- Naive clamp collapses because can't amplify
- Mean-preserving preserves because allows amplification

---

## Why This Should Work

### The Physics

**Before (simple patterns, L < d):**
```
Single head sees: [A1, A2, A3, ...] clearly separated in 40D
Attention: "Just look at previous token" (easy dot product)
No interference → No need to amplify → A_act ≈ baseline
```

**After (interleaved, L > d):**
```
Single head sees: [A1, B1, A2, B2, ...] overlapping in 40D
Attention: "Find A-stream among B-noise" (hard filtering)
Geometric crowding: 128 vectors fight for 40 dimensions
Semantic crowding: Two streams compete for representation
→ MUST amplify relevant signal → A_act >> baseline
```

### The Compensation Mechanism

1. **Constraint reduces routing** (3/4 heads frozen)
2. **Can't solve geometrically** (not enough heads to filter streams)
3. **Must solve via amplitude** (scale up signal-to-noise)
4. **Amplitude spike** (A_act increases 2-4×)
5. **Variance spike** (σ² follows as byproduct)

**Naive clamp blocks amplitude → can't compensate → collapse**
**Mean-preserving allows amplitude → can compensate → preserved**

---

## Validation Checklist

After running, verify:

### ✓ Configuration Valid
```
[CONFIG VALIDATION]
  n/d ratio: >100    ← Check
  L/d ratio: >3.0    ← Check
  OPTIMAL            ← Check
```

### ✓ Control Learns
```
Control accuracy: 60-80%  ← Task is learnable
```

### ✓ Constraint Struggles
```
Constraint accuracy: 25-40%  ← 30-50 pp DROP from control
A_act: 15-25               ← 2-4× baseline (~6-7)
Variance: 2-4× baseline     ← Allostatic load visible
```

### ✓ Clamps Dissociate
```
Naive clamp: <5% acc       ← Collapse (A blocked)
Mean-pres: ~90% of constraint ← Preserved (A allowed)
```

### ✓ Amplitude is Mechanism
```
Naive A_act ≈ baseline      ← Clamped
Mean-pres A_act ≈ constraint ← Allowed to scale
Naive fails, Mean-pres works ← A is mechanism, σ² is byproduct
```

---

## If It Still Doesn't Work

### Option 1: More Constraint
```bash
# Try d_model=32 for even tighter bottleneck
python experiments/exp_a_foundation.py --d_model 32 --vocab_size 4096 --n_epochs 50
```

### Option 2: Verify Interleaving
```python
# Quick test to check dataset generates properly
from experiments.exp_a_foundation import InterleavedSequenceDataset
ds = InterleavedSequenceDataset(10, 128, 100)
x, y = ds[0]
print(x[:10])  # Should show alternating pattern
```

### Option 3: Increase Training
```bash
# Model might need more time to learn complex task
python experiments/exp_a_foundation.py --d_model 40 --n_epochs 100 --seed 42
```

---

## The Theory

This implements the **Conservation of Separability** under dual constraint:

```
Ψ = G + A

Where:
- Ψ: Performance (must maintain to solve task)
- G: Geometric routing (frozen heads reduce this)
- A: Amplitude scaling (increases to compensate)

Under constraint:
1. G decreases (1 head << 4 heads)
2. A increases (amplitude spike)
3. Ψ maintained (performance preserved where possible)

The clamps dissociate:
- Naive blocks both A and σ² → Ψ collapses
- Mean-preserving blocks σ² only → Ψ preserved
→ Proves A is mechanism, σ² is byproduct
```

---

## Runtime

With RTX 4070 and these settings:
- **Control:** ~3-4 minutes (50 epochs)
- **Constraint:** ~3-4 minutes
- **Naive Clamp:** ~3-4 minutes
- **Mean-Preserving:** ~3-4 minutes
- **Total:** ~15-20 minutes for all 4 conditions

---

**Status: Ready to run the definitive test**

This implementation addresses both geometric (L > d) and semantic (stream interference) crowding. The constraint should now create measurable stress and trigger amplitude compensation.

```bash
# Run it!
cd clean_audit
python experiments/exp_a_foundation.py --d_model 40 --vocab_size 4096 --n_epochs 50 --seed 42
```

🎯
