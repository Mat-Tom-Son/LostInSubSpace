# LostInSubSpace: The G × S Decomposition of Transformer Robustness

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A mechanistic interpretability study proving Transformer robustness splits into Geometry (G) and Slack (S).**

---

## Key Insight

Transformer robustness decomposes into two orthogonal factors:

- **Geometry (G)**: The attention routing (QK parameters) — determines *what* representations are possible
- **Slack (S)**: The activation magnitudes (V/MLP weights) — provides *margin* for noise tolerance

**The Proof**: Swapping QK parameters between two trained models causes catastrophic failure (99% → 0%), even though both models solved the same task. This proves G *causally* constrains S.

---

## Experiments

| Phase | Question | Result |
|-------|----------|--------|
| **1** | Is G causal? | ✅ QK-swap causes 99.98% drop |
| **2** | Does G lock before S? | ✅ "Geometry Annealing" at step 4000 |
| **3** | Is S multi-dimensional? | ✅ Young G permits orthogonal S |
| **4** | Does it scale to 4L? | ⚠️ Metastable dynamics emerge |
| **5** | Language modeling (8L)? | ✅ Works, but no grokking |
| **6** | Othello world models? | 🔄 In progress (A100) |

---

## Quick Start

```bash
# Clone
git clone https://github.com/Mat-Tom-Son/LostInSubSpace.git
cd LostInSubSpace

# Install
pip install -r requirements.txt

# Run key experiment (G causality)
python clean_audit/experiments/exp_1_interleaved.py
```

---

## Repository Structure

```
LostInSubSpace/
├── clean_audit/
│   ├── lib/                    # Core utilities
│   │   ├── metrics.py          # G, S measurement
│   │   ├── clamps.py           # Sedation interventions
│   │   ├── othello_dataset.py  # Othello game simulator
│   │   └── deep_transformer.py # Multi-layer model
│   │
│   ├── experiments/
│   │   ├── phase_4/            # 4-Layer metastable dynamics
│   │   ├── phase_5/            # TinyStories (8L language)
│   │   └── phase_6/            # Othello-GPT [NEW]
│   │
│   └── data/                   # Results (gitignored)
│
├── paper/
│   └── final_report.pdf        # Full paper
│
├── FINDINGS.md                 # Detailed research log
└── README.md                   # You are here
```

---

## Phase 6: Othello-GPT (Current Focus)

Testing G × S on a **world-model task** where the model must:
1. Track a hidden board state from move sequences
2. Predict legal moves

This bridges the gap between synthetic tasks (modular arithmetic) and messy real-world tasks (language).

**Run on A100:**
```bash
cd clean_audit
bash run_cloud.sh
```

---

## Citation

```bibtex
@article{lostinsubspace2026,
  title={LostInSubSpace: The G × S Decomposition of Transformer Robustness},
  author={Thompson, Mat},
  year={2026},
  url={https://github.com/Mat-Tom-Son/LostInSubSpace}
}
```

---

## License

MIT
