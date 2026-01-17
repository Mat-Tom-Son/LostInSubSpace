"""
Part B Utilities: Parameter Freezing and Routing Manipulation

Provides utilities for:
- Freezing specific parameter groups (QK, OV, MLP)
- Extracting and swapping QK parameters between models
- Computing baseline comparison metrics
- Post-hoc suppressor measurement

These utilities enforce the G vs S boundary defined in Part B constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


def freeze_parameters(model: nn.Module, freeze_qk: bool = False, 
                     freeze_ov: bool = False, freeze_mlp: bool = False,
                     layer_filter: Optional[List[int]] = None) -> Dict[str, int]:
    """
    Freeze specific parameter groups in a transformer model.
    
    Args:
        model: The model to freeze parameters in
        freeze_qk: If True, freeze Query and Key projection parameters (Geometry)
        freeze_ov: If True, freeze Value and Output projection parameters (Slack)
        freeze_mlp: If True, freeze MLP parameters (Slack)
        layer_filter: Optional list of layer indices to apply freezing to.
                     If None, applies to all layers. Uses 'blocks.{i}' naming.
    
    Returns:
        Dictionary with counts of frozen parameters per group
        
    Note:
        - QK parameters determine routing → Geometry (G)
        - OV, MLP, LayerNorm, Embeddings → Slack (S)
    """
    frozen_counts = {'qk': 0, 'ov': 0, 'mlp': 0}
    
    def matches_layer_filter(name: str) -> bool:
        """Check if parameter name matches layer filter."""
        if layer_filter is None:
            return True
        # Check for 'blocks.{i}.' pattern
        if 'blocks.' in name:
            try:
                layer_idx = int(name.split('blocks.')[1].split('.')[0])
                return layer_idx in layer_filter
            except (ValueError, IndexError):
                return True  # Can't parse, include by default
        # Non-block parameters (embeddings, final LN) - include if no block filter
        return True
    
    for name, param in model.named_parameters():
        if not matches_layer_filter(name):
            continue
            
        # Identify parameter type (handles both 1-layer and multi-layer naming)
        is_qk = ('attn.in_proj' in name or 'attn.q_proj' in name or 'attn.k_proj' in name)
        is_ov = ('attn.v_proj' in name or 'attn.out_proj' in name)
        is_mlp = ('ff.' in name or 'ffn.' in name or 'mlp.' in name)
        
        # Apply freezing
        if freeze_qk and is_qk:
            param.requires_grad = False
            frozen_counts['qk'] += param.numel()
        elif freeze_ov and is_ov:
            param.requires_grad = False
            frozen_counts['ov'] += param.numel()
        elif freeze_mlp and is_mlp:
            param.requires_grad = False
            frozen_counts['mlp'] += param.numel()
    
    return frozen_counts


def get_qk_parameters(model: nn.Module) -> torch.Tensor:
    """
    Extract QK parameters as a flat tensor.
    
    Args:
        model: Model to extract from
        
    Returns:
        Flattened tensor of all QK parameters
    """
    qk_params = []
    
    for name, param in model.named_parameters():
        if 'attn.in_proj' in name or 'attn.q_proj' in name or 'attn.k_proj' in name:
            qk_params.append(param.data.flatten())
    
    if len(qk_params) == 0:
        raise ValueError("No QK parameters found in model")
    
    return torch.cat(qk_params)


def get_ov_parameters(model: nn.Module) -> torch.Tensor:
    """
    Extract generic Output/Value parameters as a flat tensor.
    
    Args:
        model: Model to extract from
        
    Returns:
        Flattened tensor of all OV parameters
    """
    ov_params = []
    
    for name, param in model.named_parameters():
        # Match standard transformer naming (e.g. TransformerLens or PyTorch)
        if 'attn.v_proj' in name or 'attn.out_proj' in name or 'W_V' in name or 'W_O' in name:
            ov_params.append(param.data.flatten())
    
    if len(ov_params) == 0:
        # Fallback for some implementations
        for name, param in model.named_parameters():
            if 'attn.c_proj' in name: # GPT-2 style output
                ov_params.append(param.data.flatten())
                
    if len(ov_params) == 0:
        raise ValueError("No OV parameters found in model")
    
    return torch.cat(ov_params)


def get_mlp_parameters(model: nn.Module) -> torch.Tensor:
    """
    Extract MLP parameters as a flat tensor.
    
    Args:
        model: Model to extract from
        
    Returns:
        Flattened tensor of all MLP parameters
    """
    mlp_params = []
    
    for name, param in model.named_parameters():
        if 'ff.' in name or 'ffn.' in name or 'mlp.' in name or 'w_in' in name or 'w_out' in name:
            mlp_params.append(param.data.flatten())
    
    if len(mlp_params) == 0:
        raise ValueError("No MLP parameters found in model")
    
    return torch.cat(mlp_params)


def swap_qk_parameters(target_model: nn.Module, source_model: nn.Module,
                       n_heads: Optional[int] = None,
                       layer_filter: Optional[List[int]] = None) -> Dict[str, any]:
    """
    Swap QK parameters from source model to target model.
    
    Args:
        target_model: Model to modify (gets source's QK parameters)
        source_model: Model to copy from
        n_heads: If specified, only swap this many heads (for partial swaps)
                 If None, swap all heads
        layer_filter: Optional list of layer indices to swap.
                     If None, swap all layers. Uses 'blocks.{i}' naming.
    
    Returns:
        Dictionary with swap statistics
        
    Note:
        This is a causal intervention on Geometry (G).
        If routing causally determines behavior, this should transfer properties.
    """
    swapped_params = 0
    total_heads = target_model.n_heads if hasattr(target_model, 'n_heads') else 4
    
    if n_heads is not None and n_heads > total_heads:
        raise ValueError(f"Cannot swap {n_heads} heads, model only has {total_heads}")
    
    def matches_layer_filter(name: str) -> bool:
        """Check if parameter name matches layer filter."""
        if layer_filter is None:
            return True
        if 'blocks.' in name:
            try:
                layer_idx = int(name.split('blocks.')[1].split('.')[0])
                return layer_idx in layer_filter
            except (ValueError, IndexError):
                return True
        return True
    
    for (target_name, target_param), (source_name, source_param) in zip(
        target_model.named_parameters(), source_model.named_parameters()
    ):
        if target_name != source_name:
            raise ValueError(f"Model parameter mismatch: {target_name} vs {source_name}")
        
        # Only swap QK parameters in matching layers
        is_qk = ('attn.in_proj' in target_name or 
                'attn.q_proj' in target_name or 
                'attn.k_proj' in target_name)
        
        if is_qk and matches_layer_filter(target_name):
            if n_heads is None:
                # Full swap
                target_param.data.copy_(source_param.data)
                swapped_params += target_param.numel()
            else:
                # Partial swap (first n_heads)
                # Assume parameters are organized by heads
                head_size = target_param.shape[0] // total_heads
                head_params = n_heads * head_size
                target_param.data[:head_params].copy_(source_param.data[:head_params])
                swapped_params += head_params
    
    n_layers_swapped = len(layer_filter) if layer_filter else getattr(target_model, 'n_layers', 1)
    
    return {
        'swapped_params': swapped_params,
        'n_heads_swapped': n_heads if n_heads is not None else total_heads,
        'total_heads': total_heads,
        'n_layers_swapped': n_layers_swapped
    }


def compute_baseline_metrics(model: nn.Module, baseline_model: nn.Module,
                             dataloader: torch.utils.data.DataLoader,
                             device: str = 'cpu') -> Dict[str, float]:
    """
    Compute comparison metrics against baseline model.
    
    Required metrics (as per Part B constraints):
    1. Attention pattern CosSim vs baseline
    2. QK norm drift
    3. Head-level attention entropy
    4. Residual direction similarity
    
    Args:
        model: Current model
        baseline_model: Baseline model to compare against
        dataloader: Data to evaluate on
        device: Device to run on
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    baseline_model.eval()
    
    with torch.no_grad():
        # Collect attention patterns and residuals
        model_attns = []
        baseline_attns = []
        model_resids = []
        baseline_resids = []
        
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device)
            
            # Forward pass
            _ = model(inputs)
            _ = baseline_model(inputs)
            
            # Cache should contain attention and residuals
            if 'attn_weights' in model.cache:
                model_attns.append(model.cache['attn_weights'].cpu())
                baseline_attns.append(baseline_model.cache['attn_weights'].cpu())
            
            if 'resid_post' in model.cache:
                model_resids.append(model.cache['resid_post'].cpu())
                baseline_resids.append(baseline_model.cache['resid_post'].cpu())
        
        # Concatenate
        model_attns = torch.cat(model_attns, dim=0) if model_attns else None
        baseline_attns = torch.cat(baseline_attns, dim=0) if baseline_attns else None
        model_resids = torch.cat(model_resids, dim=0) if model_resids else None
        baseline_resids = torch.cat(baseline_resids, dim=0) if baseline_resids else None
        
        metrics = {}
        
        # 1. Attention pattern similarity
        if model_attns is not None and baseline_attns is not None:
            attn_cosim = F.cosine_similarity(
                model_attns.flatten(), 
                baseline_attns.flatten(), 
                dim=0
            )
            metrics['attn_cosim_vs_baseline'] = attn_cosim.item()
            
            # 3. Head-level entropy
            # Note: PyTorch MultiheadAttention returns averaged attention weights
            # Shape is [batch, seq, seq], not [batch, heads, seq, seq]
            # We'll compute overall entropy instead of per-head
            attn_h = model_attns
            # Add small epsilon to avoid log(0)
            entropy = -(attn_h * torch.log(attn_h + 1e-10)).sum(dim=-1).mean()
            metrics['mean_attention_entropy'] = entropy.item()
            metrics['head_entropies'] = []  # Not available from averaged weights
        
        # 2. QK norm drift
        qk_current = get_qk_parameters(model)
        qk_baseline = get_qk_parameters(baseline_model)
        metrics['qk_norm_drift'] = (qk_current - qk_baseline).norm().item()
        
        # 4. Residual direction similarity
        if model_resids is not None and baseline_resids is not None:
            resid_cosim = F.cosine_similarity(
                model_resids.flatten(),
                baseline_resids.flatten(),
                dim=0
            )
            metrics['resid_direction_cosim'] = resid_cosim.item()
    
    return metrics


def measure_suppressor_strength(model: nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                device: str = 'cpu',
                                percentile: float = 0.9) -> Dict[str, float]:
    """
    Measure suppressor presence via post-hoc variance analysis.
    
    This is OBSERVATIONAL ONLY - never used in loss functions.
    
    Suppressors are defined as high-variance features in the residual stream.
    
    Args:
        model: Model to analyze
        dataloader: Data to evaluate on
        device: Device
        percentile: Threshold for identifying suppressors (default: top 10%)
        
    Returns:
        Dictionary with suppressor metrics
    """
    model.eval()
    
    with torch.no_grad():
        residuals = []
        
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device)
            
            _ = model(inputs)
            
            if 'resid_post' in model.cache:
                residuals.append(model.cache['resid_post'].cpu())
        
        if not residuals:
            return {'suppressor_strength': 0.0, 'n_suppressors': 0, 'variance_ratio': 1.0}
        
        residuals = torch.cat(residuals, dim=0)  # [n_samples, seq_len, d_model]
        
        # Compute feature-wise variance across samples
        feature_variance = residuals.var(dim=(0, 1))  # [d_model]
        
        # Identify suppressors as top percentile
        threshold = torch.quantile(feature_variance, percentile)
        suppressor_mask = feature_variance > threshold
        
        n_suppressors = suppressor_mask.sum().item()
        suppressor_strength = feature_variance[suppressor_mask].mean().item() if n_suppressors > 0 else 0.0
        variance_ratio = (feature_variance.max() / (feature_variance.median() + 1e-10)).item()
        
        return {
            'suppressor_strength': suppressor_strength,
            'n_suppressors': n_suppressors,
            'variance_ratio': variance_ratio,
            'mean_variance': feature_variance.mean().item(),
            'max_variance': feature_variance.max().item()
        }


def verify_freeze(model: nn.Module, initial_qk: torch.Tensor) -> bool:
    """
    Verify that QK parameters haven't changed (freeze actually worked).
    
    Args:
        model: Model to check
        initial_qk: Initial QK parameters (from get_qk_parameters)
        
    Returns:
        True if parameters match (freeze worked), False otherwise
    """
    current_qk = get_qk_parameters(model)
    drift = (current_qk - initial_qk).abs().max().item()
    
    # Allow tiny floating point errors
    return drift < 1e-6
