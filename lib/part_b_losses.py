"""
Part B Loss Functions

Implements the staged loss function approach:
- Primary runs: standard, noisy, label smoothing (core claims)
- Secondary runs: margin penalty (stress test)

All losses are margin-based. No entropy regularization or suppressor-targeting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def standard_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Standard cross-entropy loss (baseline).
    
    Args:
        logits: [batch, n_classes]
        targets: [batch]
        
    Returns:
        Scalar loss
    """
    return F.cross_entropy(logits, targets)


def noisy_training_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Loss for noisy training (Hardened-style from Part A).
    
    Noise is injected in the forward pass (residual stream), not in the loss.
    This is purely robustness pressure with no semantic confidence notion.
    
    Args:
        logits: [batch, n_classes]
        targets: [batch]
        
    Returns:
        Scalar loss
        
    Note:
        The noise injection happens BEFORE this loss is computed.
        See inject_noise() in the training loop.
    """
    return F.cross_entropy(logits, targets)


def label_smoothing_loss(logits: torch.Tensor, targets: torch.Tensor,
                         epsilon: float = 0.05) -> torch.Tensor:
    """
    Label smoothing loss (mild regularization).
    
    Smooths target distribution slightly to regularize margin
    without targeting confidence semantics.
    
    Args:
        logits: [batch, n_classes]
        targets: [batch]
        epsilon: Smoothing factor (default 0.05 = very mild)
        
    Returns:
        Scalar loss
    """
    n_classes = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Create smooth target distribution
    # (1 - epsilon) on correct class, epsilon / n_classes on all classes
    smooth_targets = torch.zeros_like(log_probs).scatter_(
        1, targets.unsqueeze(1), 1.0
    )
    smooth_targets = smooth_targets * (1 - epsilon) + epsilon / n_classes
    
    # Negative log likelihood of smooth targets
    loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
    
    return loss


def margin_penalty_loss(logits: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.1) -> torch.Tensor:
    """
    Margin-based penalty loss (SECONDARY RUNS ONLY).
    
    Penalizes ONLY negative margin (wasted slack in wrong direction).
    Does not penalize high confidence in general.
    
    Constraints (enforced):
    - Strictly margin-based (no entropy, no temperature)
    - Only penalizes negative margin via ReLU
    - Small coefficient (α ≤ 0.2)
    - Core claims must NOT depend on this loss
    
    Args:
        logits: [batch, n_classes]
        targets: [batch]
        alpha: Penalty coefficient (default 0.1, must be ≤ 0.2)
        
    Returns:
        Scalar loss
    """
    if alpha > 0.2:
        raise ValueError(f"alpha must be ≤ 0.2 for margin penalty, got {alpha}")
    
    # Standard cross-entropy
    ce = F.cross_entropy(logits, targets)
    
    # Compute margin: correct logit - max (other logits)
    correct_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # Mask out correct class to find max of incorrect logits
    masked_logits = logits.scatter(1, targets.unsqueeze(1), float('-inf'))
    max_incorrect = masked_logits.max(dim=1).values
    
    margin = correct_logits - max_incorrect
    
    # Penalize ONLY negative margin (not high confidence)
    # ReLU ensures we only care about cases where margin < 0
    penalty = F.relu(-margin).mean()
    
    return ce + alpha * penalty


def inject_noise(residual: torch.Tensor, noise_scale: float = 2.0,
                device: str = 'cpu') -> torch.Tensor:
    """
    Inject additive Gaussian noise into residual stream.
    
    This is used with noisy_training_loss to create the Hardened-style condition.
    
    Args:
        residual: [batch, seq_len, d_model]
        noise_scale: Standard deviation of noise
        device: Device to create noise on
        
    Returns:
        Noisy residual
    """
    noise = torch.randn_like(residual) * noise_scale
    return residual + noise


# Loss function registry for easy experiment configuration
LOSS_FUNCTIONS = {
    # Primary (core claims depend on these)
    'standard': standard_loss,
    'noisy': noisy_training_loss,
    'label_smooth': lambda l, t: label_smoothing_loss(l, t, epsilon=0.05),
    
    # Secondary (stress test / margin pressure variant)
    'margin_penalty': lambda l, t: margin_penalty_loss(l, t, alpha=0.1),
}


def get_loss_function(loss_name: str):
    """
    Get loss function by name.
    
    Args:
        loss_name: One of ['standard', 'noisy', 'label_smooth', 'margin_penalty']
        
    Returns:
        Loss function
        
    Raises:
        ValueError if loss_name not recognized
    """
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Available: {list(LOSS_FUNCTIONS.keys())}")
    
    return LOSS_FUNCTIONS[loss_name]
