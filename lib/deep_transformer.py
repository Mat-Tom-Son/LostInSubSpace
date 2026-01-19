"""
DeepTransformer: Multi-Layer Transformer for Phase 2 Experiments

Supports n_layers >= 1 with proper per-layer caching for:
- Attention weights (for geometry analysis)
- Residual streams (for amplitude/slack analysis)

Compatible with part_b_utils.py freeze/swap operations via
standard PyTorch naming: blocks.{i}.attn.in_proj_weight, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class TransformerBlock(nn.Module):
    """Single transformer block with pre-LN architecture."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        
        # Pre-LN architecture (matches paper spec)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connections.
        
        Args:
            x: [batch, seq, d_model]
            
        Returns:
            (output, attn_weights) tuple
        """
        # Pre-LN attention
        h_ln = self.ln1(x)
        attn_out, attn_weights = self.attn(h_ln, h_ln, h_ln, need_weights=True)
        x = x + attn_out
        
        # Pre-LN FFN
        h_ln = self.ln2(x)
        x = x + self.ffn(h_ln)
        
        return x, attn_weights


class DeepTransformer(nn.Module):
    """
    Multi-layer transformer supporting Phase 2 experiments.
    
    Architecture:
    - Token + positional embeddings
    - N transformer blocks (each: pre-LN attn + pre-LN FFN)
    - Final LayerNorm
    - Output projection head
    
    Caching:
    - self.cache['blocks.{i}.attn_weights']: Attention patterns per layer
    - self.cache['blocks.{i}.resid_post']: Residual after each block
    - self.cache['resid_post']: Final residual (post all blocks)
    - self.cache['resid_final']: Post-final-LN residual
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        max_seq_len: int = 128,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, self.d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Caching for metrics and interventions
        self.cache: Dict[str, torch.Tensor] = {}

        # Clamp function (optional, set externally for sedation tests)
        self.clamp_fn = None

        # Post-clamp noise scale (set externally for sedation tests)
        self.post_clamp_noise_scale = 0.0

        # Flag to enable/disable caching (disable for training to save memory)
        self.cache_activations = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Args:
            x: [batch, seq_len] token indices
            
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        # Clear cache
        self.cache = {}

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        h = self.token_emb(x) + self.pos_emb(positions)

        # Store pre-block residual (only if caching enabled)
        if self.cache_activations:
            self.cache['resid_pre'] = h.detach().clone()

        # Process through blocks
        for i, block in enumerate(self.blocks):
            h, attn_weights = block(h)

            # Cache per-layer outputs (only if caching enabled)
            if self.cache_activations:
                self.cache[f'blocks.{i}.attn_weights'] = attn_weights.detach()
                self.cache[f'blocks.{i}.resid_post'] = h.detach().clone()

        # Store final residual (post all blocks)
        if self.cache_activations:
            self.cache['resid_post'] = h.detach().clone()

        # Final layer norm
        h = self.ln_final(h)
        if self.cache_activations:
            self.cache['resid_final'] = h.detach().clone()

        # Apply clamp if set (for sedation experiments)
        if self.clamp_fn is not None:
            class MockHook:
                pass
            h = self.clamp_fn(h, MockHook())
            if self.cache_activations:
                self.cache['resid_clamped'] = h.detach().clone()

        # Apply post-clamp noise (for sedation tests)
        if self.post_clamp_noise_scale > 0:
            noise = torch.randn_like(h) * self.post_clamp_noise_scale
            h = h + noise
            if self.cache_activations:
                self.cache['post_clamp_noised'] = h.detach().clone()
        
        # Output projection
        logits = self.head(h)
        
        return logits
    
    def get_residual(self) -> Optional[torch.Tensor]:
        """Get the final residual for orthogonal loss computation."""
        return self.cache.get('resid_final', None)
    
    def freeze_attention_heads(self, heads_to_freeze: List[int], layer_idx: Optional[int] = None):
        """
        Freeze specified attention heads.
        
        Note: PyTorch MultiheadAttention doesn't support per-head freezing easily.
        This freezes the entire attention module for specified layers.
        
        Args:
            heads_to_freeze: List of head indices (for logging only with MHA)
            layer_idx: If specified, freeze only this layer. If None, freeze all layers.
        """
        if layer_idx is not None:
            layers_to_freeze = [layer_idx]
        else:
            layers_to_freeze = range(self.n_layers)
        
        frozen_count = 0
        for i in layers_to_freeze:
            for param in self.blocks[i].attn.parameters():
                param.requires_grad = False
                frozen_count += param.numel()
        
        print(f"Frozen {frozen_count} attention parameters across layers {list(layers_to_freeze)}")


class DeepModularTransformer(nn.Module):
    """
    Multi-layer transformer for modular arithmetic tasks.
    
    Similar to DeepTransformer but with:
    - Simplified positional embeddings (fixed length 2)
    - Output from last position only
    """
    
    def __init__(
        self,
        p: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        
        # Embeddings
        self.embed = nn.Embedding(p, d_model)
        self.pos_embed = nn.Embedding(2, d_model)  # Only positions 0 and 1
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, self.d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)
        
        self.cache: Dict[str, torch.Tensor] = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch, 2] - two input tokens (a, b)
            
        Returns:
            logits: [batch, p] - prediction for (a + b) mod p
        """
        B, L = x.shape
        
        self.cache = {}
        
        # Embeddings
        tok_emb = self.embed(x)
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)
        h = tok_emb + pos_emb
        
        # Process through blocks
        for i, block in enumerate(self.blocks):
            h, attn_weights = block(h)
            self.cache[f'blocks.{i}.attn_weights'] = attn_weights.detach()
            self.cache[f'blocks.{i}.resid_post'] = h.clone()
        
        # Final output from last position
        h_final = self.ln_final(h[:, -1, :])
        self.cache['resid_final'] = h_final
        self.cache['resid_post'] = h.clone()
        
        logits = self.head(h_final)
        return logits
    
    def get_residual(self) -> Optional[torch.Tensor]:
        """Get the final residual for orthogonal loss computation."""
        return self.cache.get('resid_final', None)
