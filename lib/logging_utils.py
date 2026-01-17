"""
Logging Utilities for Allostatic Load Research

Provides adaptive logging frequency, structured metric storage, and
reproducibility tools for experiments.

Key Features:
  - Dense logging during phase transitions (15-85% accuracy)
  - Sparse logging during stable phases
  - JSON-based structured storage with seed tracking
  - Checkpoint management with hash verification
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch
import numpy as np


class AuditLogger:
    """
    Centralized logging for Allostatic Load experiments.

    Handles:
      - Adaptive logging frequency
      - Structured metric storage
      - Checkpoint saving with reproducibility metadata
      - JSON export for analysis
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "data",
        seed: Optional[int] = None
    ):
        """
        Args:
            experiment_name: Name of experiment (e.g., "exp_a_control")
            output_dir: Directory for output files
            seed: Random seed for reproducibility tracking
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # Metric storage
        self.metrics_log: List[Dict[str, Any]] = []
        self.checkpoint_hashes: Dict[int, str] = {}

        # Logging state
        self.last_val_acc = 0.0
        self.log_count = 0

        # Metadata
        self.start_time = datetime.now()
        self.metadata = {
            "experiment": experiment_name,
            "seed": seed,
            "start_time": self.start_time.isoformat(),
            "git_commit": self._get_git_commit(),
        }

    def should_log(
        self,
        step: int,
        val_acc: Optional[float] = None,
        force: bool = False
    ) -> bool:
        """
        Determine if metrics should be logged at this step.

        Adaptive strategy:
          - Transition zone (15-85% accuracy): log every 10 steps
          - Stable zones (<15% or >85%): log every 100 steps
          - Always log at step 0 and multiples of 500

        Args:
            step: Current training step
            val_acc: Validation accuracy (if available)
            force: Force logging regardless of schedule

        Returns:
            True if should log at this step
        """
        if force:
            return True

        # Always log at start and major milestones
        if step == 0 or step % 500 == 0:
            return True

        # Adaptive based on validation accuracy
        if val_acc is not None:
            self.last_val_acc = val_acc
            in_transition = 0.15 < val_acc < 0.85

            if in_transition:
                # Dense logging during transition
                return step % 10 == 0
            else:
                # Sparse logging during stable phases
                return step % 100 == 0
        else:
            # Fallback: moderate frequency
            return step % 50 == 0

    def adaptive_log_frequency(self, val_acc: float, step: int) -> bool:
        """
        Convenience method matching directive specification.

        Args:
            val_acc: Validation accuracy
            step: Current step

        Returns:
            True if should log
        """
        return self.should_log(step, val_acc)

    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """
        Store metrics for this step.

        Args:
            step: Training step
            metrics: Dictionary of metric name -> value
        """
        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.metrics_log.append(record)
        self.log_count += 1

    def log_checkpoint(self, step: int, model: torch.nn.Module):
        """
        Log model checkpoint with hash for verification.

        Args:
            step: Training step
            model: Model to checkpoint
        """
        checkpoint_path = self.output_dir / f"{self.experiment_name}_step_{step}.pt"

        # Save model
        torch.save(model.state_dict(), checkpoint_path)

        # Compute hash
        checkpoint_hash = self._compute_file_hash(checkpoint_path)
        self.checkpoint_hashes[step] = checkpoint_hash

        print(f"Checkpoint saved: {checkpoint_path}")
        print(f"  Hash: {checkpoint_hash[:16]}...")

    def save_log(self, filename: Optional[str] = None):
        """
        Save complete audit log to JSON.

        Args:
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            if self.seed is not None:
                filename = f"audit_log_{self.experiment_name}_seed_{self.seed}.json"
            else:
                filename = f"audit_log_{self.experiment_name}.json"

        output_path = self.output_dir / filename

        # Prepare complete log
        complete_log = {
            "metadata": self.metadata,
            "metrics": self.metrics_log,
            "checkpoints": self.checkpoint_hashes,
            "summary": {
                "total_steps": len(self.metrics_log),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            }
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(complete_log, f, indent=2, default=str)

        print(f"Audit log saved: {output_path}")
        print(f"  Total records: {len(self.metrics_log)}")

        return output_path

    def load_log(self, filepath: str) -> Dict[str, Any]:
        """
        Load an existing audit log.

        Args:
            filepath: Path to JSON log file

        Returns:
            Complete log dictionary
        """
        with open(filepath, 'r') as f:
            log = json.load(f)

        return log

    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash if in repo."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None


class MetricAggregator:
    """
    Compute statistical summaries across runs (multiple seeds).

    Used for generating final tables with μ ± σ statistics.
    """

    def __init__(self):
        self.runs: Dict[str, List[float]] = {}

    def add_run(self, run_name: str, final_value: float):
        """
        Add final metric value from a single run.

        Args:
            run_name: Name of the metric (e.g., "accuracy", "A_param")
            final_value: Final value from this run
        """
        if run_name not in self.runs:
            self.runs[run_name] = []
        self.runs[run_name].append(final_value)

    def get_statistics(self, run_name: str) -> Dict[str, float]:
        """
        Compute statistics for a metric across runs.

        Returns:
            Dictionary with mean, std, min, max, count
        """
        if run_name not in self.runs or not self.runs[run_name]:
            return {"mean": float('nan'), "std": float('nan'), "count": 0}

        values = self.runs[run_name]
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values)
        }

    def compute_cohens_d(self, group1_name: str, group2_name: str) -> float:
        """
        Compute Cohen's d effect size between two groups.

        Args:
            group1_name: Name of first group metric
            group2_name: Name of second group metric

        Returns:
            Cohen's d (effect size)
        """
        if group1_name not in self.runs or group2_name not in self.runs:
            return float('nan')

        g1 = np.array(self.runs[group1_name])
        g2 = np.array(self.runs[group2_name])

        mean_diff = np.mean(g1) - np.mean(g2)
        pooled_std = np.sqrt((np.var(g1) + np.var(g2)) / 2)

        if pooled_std == 0:
            return float('nan')

        return mean_diff / pooled_std

    def export_summary_table(self, output_path: str):
        """
        Export summary statistics to CSV.

        Args:
            output_path: Path for output CSV file
        """
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Mean', 'Std', 'Min', 'Max', 'N'])

            for metric_name in sorted(self.runs.keys()):
                stats = self.get_statistics(metric_name)
                writer.writerow([
                    metric_name,
                    f"{stats['mean']:.4f}",
                    f"{stats['std']:.4f}",
                    f"{stats['min']:.4f}",
                    f"{stats['max']:.4f}",
                    int(stats['count'])
                ])

        print(f"Summary table saved: {output_path}")


def compare_conditions(
    log_files: Dict[str, str],
    metric_name: str,
    final_n_steps: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Compare a specific metric across multiple conditions.

    Args:
        log_files: Dict of condition_name -> log_filepath
        metric_name: Which metric to compare (e.g., "psi_accuracy")
        final_n_steps: Average over last N steps

    Returns:
        Dict of condition_name -> statistics
    """
    results = {}

    for condition_name, log_file in log_files.items():
        with open(log_file, 'r') as f:
            log = json.load(f)

        metrics = log['metrics']

        # Extract final values
        final_values = []
        for record in metrics[-final_n_steps:]:
            if metric_name in record:
                final_values.append(record[metric_name])

        if final_values:
            results[condition_name] = {
                "mean": np.mean(final_values),
                "std": np.std(final_values),
                "min": np.min(final_values),
                "max": np.max(final_values),
                "n": len(final_values)
            }
        else:
            results[condition_name] = {
                "mean": float('nan'),
                "std": float('nan'),
                "min": float('nan'),
                "max": float('nan'),
                "n": 0
            }

    return results


def extract_time_series(
    log_file: str,
    metric_names: List[str]
) -> Dict[str, List[tuple[int, float]]]:
    """
    Extract time series for specified metrics.

    Args:
        log_file: Path to audit log JSON
        metric_names: List of metrics to extract

    Returns:
        Dict of metric_name -> [(step, value), ...]
    """
    with open(log_file, 'r') as f:
        log = json.load(f)

    time_series = {name: [] for name in metric_names}

    for record in log['metrics']:
        step = record['step']
        for metric_name in metric_names:
            if metric_name in record:
                time_series[metric_name].append((step, record[metric_name]))

    return time_series


def setup_reproducibility(seed: int):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Additional CUDA reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Reproducibility setup complete: seed={seed}")


class ProgressTracker:
    """
    Simple progress tracker for training loops.

    Displays estimated time remaining and current metrics.
    """

    def __init__(self, total_steps: int, log_interval: int = 10):
        """
        Args:
            total_steps: Total number of training steps
            log_interval: How often to print progress
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.start_time = datetime.now()
        self.last_log_step = 0

    def update(self, step: int, metrics: Dict[str, float]):
        """
        Update progress and print if at log interval.

        Args:
            step: Current step
            metrics: Current metrics to display
        """
        if step % self.log_interval == 0 or step == self.total_steps - 1:
            # Compute timing
            elapsed = (datetime.now() - self.start_time).total_seconds()
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            remaining_steps = self.total_steps - step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

            # Format metrics
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

            # Print progress
            progress_pct = 100 * step / self.total_steps
            print(f"Step {step}/{self.total_steps} ({progress_pct:.1f}%) | "
                  f"ETA: {eta_seconds/60:.1f}m | {metric_str}")

            self.last_log_step = step
