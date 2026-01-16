"""
Core library modules for Allostatic Load experiments.

Modules:
    metrics: Unified observable definitions (Ψ, G, A, σ²)
    clamps: Variance clamp mechanisms
    logging_utils: Adaptive logging and reproducibility tools
    plotting: Visualization utilities
"""

from .metrics import AllostasisAudit, compute_attention_entropy
from .clamps import (
    NaiveClamp,
    MeanPreservingClamp,
    make_naive_clamp_hook,
    make_mean_preserving_clamp_hook,
    compute_baseline_statistics
)
from .logging_utils import (
    AuditLogger,
    MetricAggregator,
    setup_reproducibility,
    ProgressTracker
)

__all__ = [
    'AllostasisAudit',
    'compute_attention_entropy',
    'NaiveClamp',
    'MeanPreservingClamp',
    'make_naive_clamp_hook',
    'make_mean_preserving_clamp_hook',
    'compute_baseline_statistics',
    'AuditLogger',
    'MetricAggregator',
    'setup_reproducibility',
    'ProgressTracker',
]
