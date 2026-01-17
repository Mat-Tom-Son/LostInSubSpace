"""
Unified Metrics Module for Allostatic Load Research

Implements all observable definitions for Ψ = G + A hypothesis testing.

Observables:
  - Ψ (Separability): Functional performance (accuracy, logit_diff)
  - G (Geometry): Routing plasticity, directional alignment
  - A (Amplitude): Mean-shift magnitude in signal direction
  - σ² (Variance): Statistical dispersion (byproduct)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


class AllostasisAudit:
    """
    Unified metric computation for Allostatic Load experiments.

    Core equation: Ψ = G + A

    All methods return scalar values for consistent logging and analysis.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.attention_history = {}  # For tracking JSD over time

    def compute_psi_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute Ψ (Separability) as classification accuracy.

        Args:
            logits: Model output logits [batch, seq, vocab] or [batch, vocab]
            labels: Ground truth labels [batch, seq] or [batch]

        Returns:
            Accuracy as float in [0, 1]
        """
        with torch.no_grad():
            if logits.dim() == 3:
                # Sequence prediction: take last token or flatten
                logits = logits[:, -1, :]
                if labels.dim() == 2:
                    labels = labels[:, -1]

            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == labels).float().mean().item()

        return accuracy

    def compute_psi_logit_diff(
        self,
        logits: torch.Tensor,
        correct_idx: int,
        incorrect_idx: int
    ) -> float:
        """
        Compute Ψ (Separability) as logit difference.

        Useful for IOI and other tasks with specific target tokens.

        Args:
            logits: Model output logits [batch, vocab] or [batch, seq, vocab]
            correct_idx: Index of correct token
            incorrect_idx: Index of incorrect token

        Returns:
            log(P(correct)) - log(P(incorrect))
        """
        with torch.no_grad():
            if logits.dim() == 3:
                logits = logits[:, -1, :]  # Take last position

            log_probs = F.log_softmax(logits, dim=-1)
            logit_diff = (log_probs[:, correct_idx] - log_probs[:, incorrect_idx]).mean().item()

        return logit_diff

    def compute_geometry_jsd(
        self,
        attention_pattern: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        step: int,
        window: int = 500
    ) -> float:
        """
        Compute G (Geometry / Routing Plasticity) via JSD of attention patterns.

        G = 1 - JSD(attn_t, attn_{t-τ})

        Higher G → more plasticity (patterns still changing)
        Lower G → frozen routing (patterns settled)

        Args:
            attention_pattern: [batch, n_heads, seq_q, seq_k] or [seq_q, seq_k]
            layer_idx: Layer index
            head_idx: Head index within layer
            step: Current training step
            window: Time window τ for comparison

        Returns:
            G value in [0, 1]. High = plastic, Low = frozen.
        """
        # Ensure 2D: [seq_q, seq_k]
        if attention_pattern.dim() == 4:
            attn = attention_pattern[0, head_idx]  # Take first batch, specific head
        elif attention_pattern.dim() == 3:
            attn = attention_pattern[head_idx]
        else:
            attn = attention_pattern

        # Flatten to probability distribution
        attn_flat = attn.flatten().cpu().numpy()
        attn_flat = attn_flat / (attn_flat.sum() + 1e-10)  # Normalize

        key = f"L{layer_idx}H{head_idx}"

        # Store current pattern
        if key not in self.attention_history:
            self.attention_history[key] = []

        self.attention_history[key].append({
            'step': step,
            'pattern': attn_flat
        })

        # Compute JSD with previous pattern
        history = self.attention_history[key]
        if len(history) < 2:
            return 1.0  # Maximum plasticity at start

        # Find pattern from window steps ago
        prev_pattern = None
        for record in reversed(history[:-1]):
            if step - record['step'] >= window:
                prev_pattern = record['pattern']
                break

        if prev_pattern is None:
            # Use oldest available
            prev_pattern = history[0]['pattern']

        # Compute Jensen-Shannon Divergence
        jsd = jensenshannon(attn_flat, prev_pattern)

        # G = 1 - JSD (invert so high G = high plasticity)
        # But actually, the directive says low G = frozen, high G = plastic
        # And JSD increases with divergence, so:
        # High JSD → high change → high plasticity → should be high G
        # Let's use G = 1 - JSD so that:
        # - When patterns are the same: JSD=0, G=1 (highly plastic/not frozen yet)
        # - When patterns differ: JSD=high, G=low (frozen into different pattern)
        #
        # Wait, re-reading: "G ∈ [0, 1]. High G = routing still exploring. Low G = frozen."
        # And "1 - (attention pattern drift over time window)"
        # So drift = JSD, and G = 1 - drift
        # High drift → Low G (frozen into new pattern)
        # Low drift → High G (still exploring, not settled)

        G = 1.0 - jsd
        G = np.clip(G, 0.0, 1.0)

        return float(G)

    def compute_amplitude_learned(self, model: nn.Module) -> float:
        """
        Compute A_learned (Amplitude / Learnable Gain).

        Definition: Mean scale parameter of final LayerNorm.

        This is the gradient-descent-tunable parameter.

        Args:
            model: Transformer model (HookedTransformer or similar)

        Returns:
            Mean of ln_final.w (scale parameter)
        """
        try:
            # TransformerLens: model.ln_final.w
            if hasattr(model, 'ln_final') and hasattr(model.ln_final, 'w'):
                A_learned = model.ln_final.w.data.mean().item()
            # Standard PyTorch: model.ln_final.weight
            elif hasattr(model, 'ln_final') and hasattr(model.ln_final, 'weight'):
                A_learned = model.ln_final.weight.data.mean().item()
            # Look for any final norm layer
            elif hasattr(model, 'norm') and hasattr(model.norm, 'weight'):
                A_learned = model.norm.weight.data.mean().item()
            else:
                # Fallback: search for last LayerNorm
                last_ln = None
                for module in model.modules():
                    if isinstance(module, nn.LayerNorm):
                        last_ln = module
                if last_ln is not None and hasattr(last_ln, 'weight'):
                    A_learned = last_ln.weight.data.mean().item()
                else:
                    raise AttributeError("Could not find final LayerNorm in model")

            return A_learned

        except Exception as e:
            print(f"Warning: Could not compute A_learned: {e}")
            return float('nan')

    def compute_amplitude_activation(self, resid: torch.Tensor) -> float:
        """
        Compute A_activation (Amplitude / Activation Volume).

        Definition: Global mean residual norm (pre-LN) at layer ℓ.

        Args:
            resid: Residual stream activations [batch, seq, d_model]

        Returns:
            Scalar mean residual norm
        """
        with torch.no_grad():
            # Flatten to [batch*seq, d_model]
            resid_flat = resid.reshape(-1, resid.shape[-1])

            # Compute L2 norm per token, then mean across all tokens
            norms = resid_flat.norm(dim=-1)
            A_activation = norms.mean().item()

        return A_activation

    def compute_amplitude_param(self, model: nn.Module, param_name: str = "W_V") -> float:
        """
        Compute A_param (Amplitude / Parameter-Level Cost).

        Definition: L2 norm of specified projection weights (default: W_V only).

        Args:
            model: Transformer model
            param_name: Name of parameter to track (e.g., "W_V", "W_O")

        Returns:
            Sum of parameter norms across layers
        """
        total_norm = 0.0
        count = 0

        for name, param in model.named_parameters():
            if param_name in name and 'attn' in name:
                total_norm += param.norm().item()
                count += 1

        if count == 0:
            # Fallback: try different naming conventions
            for name, param in model.named_parameters():
                # TransformerLens: blocks.{i}.attn.W_V
                # HuggingFace: layer.{i}.attention.value.weight
                if any(kw in name.lower() for kw in ['value', 'w_v', 'v_proj']):
                    if 'attn' in name.lower() or 'attention' in name.lower():
                        total_norm += param.norm().item()
                        count += 1

        return total_norm

    def compute_variance(self, resid: torch.Tensor) -> float:
        """
        Compute σ²_resid (Variance / Dispersion).

        Definition: Variance of residual activations (post-centering).

        Important: This is a BYPRODUCT, not the mechanism.
        The mechanism is A (mean shift), variance is the symptom.

        Args:
            resid: Residual stream activations [batch, seq, d_model]

        Returns:
            Scalar variance (σ²)
        """
        with torch.no_grad():
            # Flatten to [batch*seq, d_model]
            resid_flat = resid.reshape(-1, resid.shape[-1])

            # Compute standard deviation per dimension, mean across tokens and dims
            sigma = resid_flat.std(dim=-1).mean().item()
            sigma_sq = sigma ** 2

        return sigma_sq

    def compute_all_metrics(
        self,
        model: nn.Module,
        logits: torch.Tensor,
        labels: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        step: int,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics at once for efficient logging.

        Args:
            model: Transformer model
            logits: Model output
            labels: Ground truth
            cache: Activation cache from forward pass
            step: Current training step
            layer_indices: Which layers to compute metrics for (default: all)

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        # Ψ (Separability)
        metrics['psi_accuracy'] = self.compute_psi_accuracy(logits, labels)

        # A_learned (global, from final LN)
        metrics['A_learned'] = self.compute_amplitude_learned(model)

        # A_param (global, sum across layers)
        metrics['A_param'] = self.compute_amplitude_param(model, "W_V")

        # Per-layer metrics
        if layer_indices is None:
            # Auto-detect number of layers
            layer_indices = []
            for key in cache.keys():
                if 'resid_post' in str(key):
                    # Extract layer number
                    if isinstance(key, tuple):
                        layer_indices.append(key[1])
                    elif 'blocks.' in str(key):
                        try:
                            layer_num = int(str(key).split('blocks.')[1].split('.')[0])
                            layer_indices.append(layer_num)
                        except:
                            pass
            layer_indices = sorted(set(layer_indices))

        for layer_idx in layer_indices:
            # Get residual stream
            resid_key = f'blocks.{layer_idx}.hook_resid_post'
            if resid_key in cache:
                resid = cache[resid_key]
            elif ('resid_post', layer_idx) in cache:
                resid = cache[('resid_post', layer_idx)]
            else:
                continue

            # A_activation (per layer)
            metrics[f'A_activation_L{layer_idx}'] = self.compute_amplitude_activation(resid)

            # σ² (per layer)
            metrics[f'sigma_sq_L{layer_idx}'] = self.compute_variance(resid)

            # G (Geometry) - requires attention patterns
            attn_key = f'blocks.{layer_idx}.attn.hook_pattern'
            if attn_key in cache:
                attn_pattern = cache[attn_key]
            elif ('pattern', layer_idx) in cache:
                attn_pattern = cache[('pattern', layer_idx)]
            else:
                attn_pattern = None

            if attn_pattern is not None and attn_pattern.dim() >= 3:
                # Compute G for each head
                n_heads = attn_pattern.shape[1] if attn_pattern.dim() == 4 else attn_pattern.shape[0]
                for head_idx in range(n_heads):
                    G = self.compute_geometry_jsd(
                        attn_pattern,
                        layer_idx,
                        head_idx,
                        step,
                        window=500
                    )
                    metrics[f'G_L{layer_idx}H{head_idx}'] = G

        return metrics

    def reset_attention_history(self):
        """Clear attention pattern history (e.g., between experiments)."""
        self.attention_history = {}


def compute_attention_entropy(attention_pattern: torch.Tensor) -> float:
    """
    Compute entropy of attention distribution.

    Higher entropy → more uniform attention (less focused)
    Lower entropy → peaked attention (more focused)

    Args:
        attention_pattern: [batch, n_heads, seq_q, seq_k] or [n_heads, seq_q, seq_k]

    Returns:
        Mean entropy across heads and positions
    """
    with torch.no_grad():
        # Ensure we have [n_heads, seq_q, seq_k]
        if attention_pattern.dim() == 4:
            attn = attention_pattern[0]  # Take first batch
        else:
            attn = attention_pattern

        # Compute entropy for each head and query position
        entropies = []
        for head_idx in range(attn.shape[0]):
            for q_idx in range(attn.shape[1]):
                dist = attn[head_idx, q_idx].cpu().numpy()
                dist = dist / (dist.sum() + 1e-10)  # Normalize
                ent = entropy(dist)
                entropies.append(ent)

        mean_entropy = np.mean(entropies)

    return float(mean_entropy)


def compute_mean_resid_norm(
    model: nn.Module,
    batch: torch.Tensor,
    layer_idx: Optional[int] = None
) -> float:
    """
    Convenience function to compute mean residual norm.

    Args:
        model: Transformer model
        batch: Input batch
        layer_idx: Specific layer (if None, returns final layer)

    Returns:
        Mean residual norm
    """
    try:
        # This requires the model to support caching
        if hasattr(model, 'run_with_cache'):
            _, cache = model.run_with_cache(batch)
        else:
            # Standard forward pass
            with torch.no_grad():
                _ = model(batch)
                # Would need hooks to capture - simplified for now
                return float('nan')

        # Get residual stream
        if layer_idx is not None:
            resid_key = f'blocks.{layer_idx}.hook_resid_post'
        else:
            # Final layer
            resid_key = f'blocks.{model.cfg.n_layers-1}.hook_resid_post' if hasattr(model, 'cfg') else 'ln_final'

        if resid_key in cache:
            resid = cache[resid_key]
        else:
            return float('nan')

        auditor = AllostasisAudit()
        return auditor.compute_amplitude_activation(resid)

    except Exception as e:
        print(f"Warning: Could not compute mean resid norm: {e}")
        return float('nan')


def compute_resid_variance(
    model: nn.Module,
    batch: torch.Tensor,
    layer_idx: Optional[int] = None
) -> float:
    """
    Convenience function to compute residual variance.

    Args:
        model: Transformer model
        batch: Input batch
        layer_idx: Specific layer (if None, returns final layer)

    Returns:
        Variance (σ²)
    """
    try:
        # Similar to compute_mean_resid_norm
        if hasattr(model, 'run_with_cache'):
            _, cache = model.run_with_cache(batch)
        else:
            return float('nan')

        if layer_idx is not None:
            resid_key = f'blocks.{layer_idx}.hook_resid_post'
        else:
            resid_key = f'blocks.{model.cfg.n_layers-1}.hook_resid_post' if hasattr(model, 'cfg') else 'ln_final'

        if resid_key in cache:
            resid = cache[resid_key]
        else:
            return float('nan')

        auditor = AllostasisAudit()
        return auditor.compute_variance(resid)

    except Exception as e:
        print(f"Warning: Could not compute variance: {e}")
        return float('nan')
