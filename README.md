# The Conservation of Separability: Allostatic Load in Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A mechanistic interpretability study validating the G × S decomposition of Transformer robustness.**

---

## Abstract

We demonstrate that Transformer robustness decomposes into two orthogonal, causally distinct factors:

- **Geometry (G)**: Attention routing patterns (QK parameters) that determine representational affordances
- **Slack (S)**: Multidimensional margin allocation in residual space (V/MLP weights) that provides robustness

**Key Finding**: Transplanting attention routing between converged models causes catastrophic failure (99.99% → 0.02%), proving that G causally constrains behavior and S is critically G-dependent.

---

## Repository Structure

```
allo-audit/
├── README.md              # You are here
├── FINDINGS.md            # Complete research findings
├── requirements.txt       # Dependencies
│
├── clean_audit/           # Core codebase
│   ├── lib/               # Library modules
│   │   ├── metrics.py     # Ψ, G, A, σ² computation
│   │   ├── clamps.py      # Variance clamping interventions
│   │   ├── part_b_utils.py # QK freezing/swapping utilities
│   │   └── ...
│   │
│   ├── experiments/       # Runnable experiment scripts
│   │   ├── exp_a_foundation.py    # Part A: Necessity of slack
│   │   ├── exp_1_interleaved.py   # Part B: G causality (routing swap)
│   │   ├── exp_2_interleaved.py   # Part B: Temporal ordering
│   │   ├── exp_3_interleaved.py   # Part B: S multidimensionality
│   │   ├── phase_2/               # 2-Layer scaling validation
│   │   └── phase_4/               # 4-Layer metastable dynamics
│   │
│   └── data/              # Experiment results (JSON)
│
├── paper/                 # LaTeX source and figures
│   ├── final_report.tex
│   └── final_report.pdf
│
├── checkpoints/           # Model weights (.pt files)
├── scripts/               # Utility scripts
├── archive/               # Historical documentation
└── docs/                  # Extended documentation
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/allo-audit.git
cd allo-audit

# Install dependencies
pip install -r requirements.txt
```

### Run the Key Experiment (G Causality)

```bash
# Run Experiment 1: Routing Swap on Interleaved Task
python clean_audit/experiments/exp_1_interleaved.py --device cuda --n_steps 10000
```

**Expected Output**: Two models train to 99.99% accuracy, then swapping QK parameters causes accuracy to drop to 0.02%, validating G causality.

---

## Experiments Overview

| Experiment | Question | Result |
|------------|----------|--------|
| **Exp A** | Is slack necessary? | High-precision tasks accumulate prophylactic margin |
| **Exp 1** | Is G causal? | Yes - swap causes 99.98% accuracy drop |
| **Exp 2B** | Does G lock before S? | Yes - QK freezes at Step 4000 (Annealing) |
| **Exp 3** | Is S multidimensional? | Yes - Young G permits orthogonal S allocations |
| **Phase 2** | Does G×S scale to 2L? | Yes - All findings replicate at 2 layers |
| **Phase 4** | What happens at 4L? | **Metastable dynamics** - stability is stochastic |
| **Phase 5** | Does G×S hold on language? | ✅ **Yes** - Ortho works, but no grokking (task too easy) |

---

## Key Results

### Part A: Prophylactic Amplitude
- **Naive clamping** (block all variance): 0% accuracy
- **Mean-preserving clamping** (block variance, keep mean): 0% accuracy
- **Conclusion**: Information is in the variance distribution, not just the mean

### Part B: Causal Validation
- **Routing Swap** (Exp 1): Swapping QK between converged models causes catastrophic failure
- **S Collapse** (Exp 3): Different training objectives converge to same residual direction under frozen G
- **Conclusion**: G and S are separable but causally interdependent

### Phase 4: Depth-Dependent Dynamics (4-Layer)

> **Key Discovery**: Stability at depth is not guaranteed by training objectives; it emerges probabilistically from a metastable regime.

| Finding | Evidence |
|---------|----------|
| **Metastable oscillations** | Models collapse/recover repeatedly before settling |
| **Stochastic escape** | 40-50% stability rate across seeds |
| **λ sweep falsification** | λ=0 (no ortho) has HIGHEST stability (50% vs 25%) |
| **Loss term engineering fails** | No simple penalty reliably drives escape |

**Implication**: The phenomenon is about landscape topology, not optimization.

### Phase 5: Natural Language Scaling (8-Layer TinyStories) ✅

> **Complete**: G×S decomposition validated on language modeling at 8 layers.

| Finding | Result |
|---------|--------|
| **Ortho mechanism works** | ✅ CosSim 1.0 → 0.001 while maintaining 99.4% accuracy |
| **100% stability** | ✅ No collapses across 3 seeds × 4 conditions |
| **No grokking observed** | TinyStories too easy - smooth convergence |

**Implication**: Orthogonality mechanism generalizes to language, but metastability requires algorithmic tasks.

---

## Developer's Guide & Learned Insights

Insights for future researchers inspecting these models:

1. **Model Inspection**:
   - Use `debug_model.py` to inspect logits and loss on specific inputs.
   - **Key Checkpoint**: `checkpoints/healthy_victim_modular.pt` (High-precision model, p=113).
   - *Tip*: When debugging "collapse", check **Cosine Similarity** of residuals first. Accuracy often stays high (99%) while internal representations drift (CosSim < 0.5), until sudden failure.

2. **Reproduction Tools**:
   - **Sedation Curve**: `clean_audit/experiments/exp_sedation_curve.py`
     - Runs the "Clamp Sweep" to prove margin is a noise buffer.
     - *Artifact*: `paper/sedation_curve.png`.
   - **Drift Tracking**: `clean_audit/experiments/exp_drift_tracking.py`
     - Tracks parameter velocity ($v_t$) to visualize "Geometry Annealing".
     - *Artifact*: `paper/drift_tracking.png`.

3. **Architectural Findings**:
   - **Interleaved vs Modular**: The Interleaved task (d=64) is "cramped" and fragile (~15% acc under severe noise). The Modular task (d=128) allows massive prophylactic margins (~80% acc under severe noise). Robustness requires **geometric space**.
   - **No Gain Reflex**: Transformers are passive at inference. They do not "react" to noise by increasing gain. Robustness is strictly **pre-allocated slack**.

---

## Citation

```bibtex
@article{allostatic2026,
  title={The Conservation of Separability: Allostatic Load in Transformers},
  author={Research Team},
  year={2026}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
