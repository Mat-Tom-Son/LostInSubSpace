"""
Variance Clamp Mechanisms for Allostatic Load Research

Implements two clamp types to dissociate amplitude (A) from variance (σ²):

1. Naive Clamp: Blocks BOTH mean shift and variance
   → Prediction: Catastrophic collapse (proves both needed)

2. Mean-Preserving Clamp: Blocks variance only, preserves mean shift
   → Prediction: Performance maintained (proves A is mechanism, σ² is byproduct)

Critical Design:
- Clamps operate during FORWARD PASS (gradients flow through clamped space)
- Clamps are implemented as PyTorch hooks
- Reference values (target_norm, healthy_std) are fixed from baseline runs
"""

import torch
import torch.nn as nn
from typing import Optional, Callable


class NaiveClamp:
    """
    Naive Variance Clamp: Blocks both amplitude scaling and variance.

    Mechanism:
        resid_clamped = resid * (TARGET_NORM / current_norm)

    This rescales the entire residual stream to match a healthy baseline norm.
    It constrains BOTH:
        - Mean shift in signal direction (A)
        - Variance dispersion (σ²)

    Expected result: Catastrophic performance collapse when combined with
    architectural constraints (e.g., frozen attention heads).
    """

    def __init__(self, target_norm: float, layer_idx: int):
        """
        Args:
            target_norm: Fixed norm from healthy baseline (scalar)
            layer_idx: Which layer to apply clamp to
        """
        self.target_norm = target_norm
        self.layer_idx = layer_idx
        self.apply_count = 0

    def __call__(self, resid: torch.Tensor, hook) -> torch.Tensor:
        """
        Hook function to apply naive clamp.

        Args:
            resid: Residual stream [batch, seq, d_model]
            hook: Hook object (from TransformerLens)

        Returns:
            Clamped residual with target norm
        """
        with torch.enable_grad():  # Allow gradients to flow
            # Flatten to [batch*seq, d_model]
            original_shape = resid.shape
            resid_flat = resid.reshape(-1, resid.shape[-1])

            # Compute current mean norm
            current_norm = resid_flat.norm(dim=-1).mean()

            # Avoid division by zero
            if current_norm < 1e-8:
                return resid

            # Rescale to target norm
            scale_factor = self.target_norm / current_norm
            resid_clamped = resid * scale_factor
            
            # DEBUG: Verify clamp is working
            clamped_flat = resid_clamped.reshape(-1, resid_clamped.shape[-1])
            clamped_norm = clamped_flat.norm(dim=-1).mean()
            
            if self.apply_count < 3:  # Only print first few times
                print(f"  [CLAMP DEBUG] Before: {current_norm:.4f} -> After: {clamped_norm:.4f} (target: {self.target_norm:.4f}, scale: {scale_factor:.4f})")

            self.apply_count += 1

            return resid_clamped

    def get_hook_name(self) -> str:
        """Return the hook name for this layer."""
        return f"blocks.{self.layer_idx}.hook_resid_post"

    def reset(self):
        """Reset application counter."""
        self.apply_count = 0


class MeanPreservingClamp:
    """
    Mean-Preserving Variance Clamp: Blocks variance only, allows amplitude scaling.

    Mechanism:
        1. Center: centered = resid - mean
        2. Clamp std: clamped_centered = centered / current_std * HEALTHY_STD
        3. Restore mean: resid_clamped = mean + clamped_centered

    This constrains ONLY:
        - Variance dispersion (σ²)

    While allowing:
        - Mean shift in signal direction (A)

    Expected result: Performance preserved at ~90% of constrained baseline,
    proving that amplitude (A) is the compensatory mechanism, not variance.
    """

    def __init__(self, healthy_std: float, layer_idx: int):
        """
        Args:
            healthy_std: Fixed std from healthy baseline (scalar)
            layer_idx: Which layer to apply clamp to
        """
        self.healthy_std = healthy_std
        self.layer_idx = layer_idx
        self.apply_count = 0

    def __call__(self, resid: torch.Tensor, hook) -> torch.Tensor:
        """
        Hook function to apply mean-preserving clamp.

        Args:
            resid: Residual stream [batch, seq, d_model]
            hook: Hook object (from TransformerLens)

        Returns:
            Variance-clamped residual with preserved mean
        """
        with torch.enable_grad():  # Allow gradients to flow
            # Flatten to [batch*seq, d_model]
            original_shape = resid.shape
            resid_flat = resid.reshape(-1, resid.shape[-1])

            # 1. Compute and preserve mean vector
            mean_vec = resid_flat.mean(dim=0, keepdim=True)

            # 2. Center the residuals
            centered = resid_flat - mean_vec

            # 3. Compute current standard deviation (scalar across all dimensions)
            current_std = centered.std()

            # Avoid division by zero
            if current_std < 1e-8:
                return resid

            # 4. Rescale centered residuals to healthy std
            clamped_centered = centered / current_std * self.healthy_std

            # 5. Restore mean (this is the key: we preserve amplitude!)
            resid_clamped_flat = mean_vec + clamped_centered

            # 6. Reshape back to original
            resid_clamped = resid_clamped_flat.reshape(original_shape)

            self.apply_count += 1

            return resid_clamped

    def get_hook_name(self) -> str:
        """Return the hook name for this layer."""
        return f"blocks.{self.layer_idx}.hook_resid_post"

    def reset(self):
        """Reset application counter."""
        self.apply_count = 0


def make_naive_clamp_hook(target_norm: float, layer_idx: int) -> Callable:
    """
    Factory function to create a naive clamp hook.

    Args:
        target_norm: Fixed norm from healthy baseline
        layer_idx: Layer to apply clamp

    Returns:
        Callable hook function
    """
    clamp = NaiveClamp(target_norm, layer_idx)
    return clamp


def make_mean_preserving_clamp_hook(healthy_std: float, layer_idx: int) -> Callable:
    """
    Factory function to create a mean-preserving clamp hook.

    Args:
        healthy_std: Fixed std from healthy baseline
        layer_idx: Layer to apply clamp

    Returns:
        Callable hook function
    """
    clamp = MeanPreservingClamp(healthy_std, layer_idx)
    return clamp


class ClampCalibrator:
    """
    Helper class to compute reference values from baseline runs.

    Usage:
        1. Train a control model (no constraints)
        2. Run ClampCalibrator to compute target_norm and healthy_std
        3. Use these values in clamped training runs
    """

    def __init__(self):
        self.norms = []
        self.stds = []

    def accumulate(self, resid: torch.Tensor):
        """
        Accumulate statistics from residual streams.

        Args:
            resid: Residual stream [batch, seq, d_model]
        """
        with torch.no_grad():
            resid_flat = resid.reshape(-1, resid.shape[-1])

            # Compute norm
            norm = resid_flat.norm(dim=-1).mean().item()
            self.norms.append(norm)

            # Compute std (mean-preserving)
            mean_vec = resid_flat.mean(dim=0, keepdim=True)
            centered = resid_flat - mean_vec
            std = centered.std().item()
            self.stds.append(std)

    def get_target_norm(self) -> float:
        """Get mean norm for naive clamp."""
        return sum(self.norms) / len(self.norms) if self.norms else 1.0

    def get_healthy_std(self) -> float:
        """Get mean std for mean-preserving clamp."""
        return sum(self.stds) / len(self.stds) if self.stds else 1.0

    def reset(self):
        """Clear accumulated statistics."""
        self.norms = []
        self.stds = []


def compute_baseline_statistics(
    model: nn.Module,
    dataloader,
    layer_idx: int = 0,
    n_batches: int = 100
) -> tuple[float, float]:
    """
    Compute baseline norm and std from a healthy model.

    Args:
        model: Trained baseline model
        dataloader: Data to compute statistics over
        layer_idx: Which layer to compute for
        n_batches: How many batches to average over

    Returns:
        (target_norm, healthy_std) tuple
    """
    calibrator = ClampCalibrator()
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            # Forward pass with cache
            if hasattr(model, 'run_with_cache'):
                _, cache = model.run_with_cache(batch)
                resid_key = f"blocks.{layer_idx}.hook_resid_post"
                if resid_key in cache:
                    resid = cache[resid_key]
                    calibrator.accumulate(resid)
            else:
                # Would need to set up hooks manually
                print("Warning: Model does not support run_with_cache")
                break

    target_norm = calibrator.get_target_norm()
    healthy_std = calibrator.get_healthy_std()

    print(f"Baseline statistics for layer {layer_idx}:")
    print(f"  Target norm: {target_norm:.4f}")
    print(f"  Healthy std: {healthy_std:.4f}")

    return target_norm, healthy_std


class AdaptiveClamp:
    """
    Experimental: Adaptive clamp that adjusts target over time.

    NOT USED in the main experiments (which use fixed targets),
    but included for future exploration.
    """

    def __init__(
        self,
        initial_target: float,
        layer_idx: int,
        decay_rate: float = 0.999,
        clamp_type: str = "naive"
    ):
        """
        Args:
            initial_target: Starting target value
            layer_idx: Layer to apply clamp
            decay_rate: Exponential decay for target adaptation
            clamp_type: "naive" or "mean_preserving"
        """
        self.target = initial_target
        self.layer_idx = layer_idx
        self.decay_rate = decay_rate
        self.clamp_type = clamp_type
        self.step = 0

    def __call__(self, resid: torch.Tensor, hook) -> torch.Tensor:
        """Apply adaptive clamp."""
        # Update target
        self.target *= self.decay_rate
        self.step += 1

        # Apply appropriate clamp
        if self.clamp_type == "naive":
            clamp = NaiveClamp(self.target, self.layer_idx)
        else:
            clamp = MeanPreservingClamp(self.target, self.layer_idx)

        return clamp(resid, hook)


# Convenience functions for quick setup

def setup_naive_clamp(
    model: nn.Module,
    target_norm: float,
    layer_idx: int = 0
) -> tuple[str, Callable]:
    """
    Quick setup for naive clamp.

    Returns:
        (hook_name, hook_function) tuple for use with run_with_hooks
    """
    clamp = make_naive_clamp_hook(target_norm, layer_idx)
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    return hook_name, clamp


def setup_mean_preserving_clamp(
    model: nn.Module,
    healthy_std: float,
    layer_idx: int = 0
) -> tuple[str, Callable]:
    """
    Quick setup for mean-preserving clamp.

    Returns:
        (hook_name, hook_function) tuple for use with run_with_hooks
    """
    clamp = make_mean_preserving_clamp_hook(healthy_std, layer_idx)
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    return hook_name, clamp
