"""
Phase 4: 4-Layer Scaling Validation

Experiments to validate G×S decomposition at 4 layers.

Experiments:
- exp_4_1_swap_4layer: Routing swap on 4-layer models
- exp_4_2_young_g_4layer: Young G subspace probe (4-layer)
- exp_4_3_sedation_4layer: Sedation test (4-layer)

Run all experiments:
    python -m clean_audit.experiments.phase_4.run_all

Or individual experiments:
    python clean_audit/experiments/phase_4/exp_4_1_swap_4layer.py
    python clean_audit/experiments/phase_4/exp_4_2_young_g_4layer.py
    python clean_audit/experiments/phase_4/exp_4_3_sedation_4layer.py
"""

from .exp_4_1_swap_4layer import run_experiment as run_swap_experiment
from .exp_4_2_young_g_4layer import run_experiment as run_young_g_experiment
from .exp_4_3_sedation_4layer import run_experiment as run_sedation_experiment

__all__ = [
    'run_swap_experiment',
    'run_young_g_experiment',
    'run_sedation_experiment'
]
