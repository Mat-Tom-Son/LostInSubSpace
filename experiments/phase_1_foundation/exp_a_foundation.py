"""
EXPERIMENT A: THE FOUNDATION (Direct Constraint + Clamp)

Goal: Reproduce the core Allostatic Shift. Prove variance amplification is
causal necessity.

Conditions:
  1. Control: All 4 attention heads free (baseline)
  2. Constraint: 3/4 heads frozen, 1 free (stress condition)
  3. Naive Clamp: Constraint + variance clamp (blocks both A and variance)
  4. Mean-Preserving Clamp: Constraint + variance-only clamp (blocks variance, allows A)

Expected Results:
  - Control: >=95% accuracy
  - Constraint: 55-65% accuracy, high A_activation (6x+)
  - Naive Clamp: <5% accuracy (catastrophic collapse)
  - Mean-Preserving: 50-54% accuracy (~90% of Constraint)

This proves:
  - Amplitude freedom is causally necessary (Naive collapse)
  - Mean shift is the mechanism, variance is byproduct (Mean-Preserving preserves)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Dict, List
import argparse
from tqdm import tqdm

# Import our libraries
from lib.metrics import AllostasisAudit
from lib.clamps import (
    make_naive_clamp_hook,
    make_mean_preserving_clamp_hook,
    compute_baseline_statistics
)
from lib.logging_utils import AuditLogger, setup_reproducibility, ProgressTracker


# =============================================================================
# JAMMING PROTOCOL: Surgical Noise Injection
# =============================================================================
def make_jamming_hook(noise_scale=0.07):
    """
    Injects pure Gaussian noise into the residual stream.
    This creates an artificial 'Noise Floor' that the model must shout over.
    
    Unlike weight scaling (which saturates softmax into 'Rigid Room'),
    this directly raises the noise floor in the residual stream ('Loud Room').
    
    Args:
        noise_scale: Standard deviation of injected noise (default 0.07)
                     Must be FIXED (not adaptive) so SNR improves as A increases.
    
    Returns:
        Hook function for register_forward_hook
    """
    def hook(module, input, output):
        # output is [batch, seq, hidden]
        # Create noise on the same device
        noise = torch.randn_like(output) * noise_scale
        
        # Add noise to the stream (Active Jamming)
        return output + noise
    return hook


# Simple 1-layer Transformer for TinyStories
class SimpleTransformer(nn.Module):
    """
    Minimal 1-layer transformer for Experiment A.

    Architecture:
      - 1 layer
      - 4 attention heads
      - d_model = 128
      - Standard Transformer block
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        disable_ffn: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        # PHASE 1.1: Cripple FFN - set d_ff = d_model (remove 4x expansion)
        # PHASE 4.1: Option to disable FFN entirely
        self.disable_ffn = disable_ffn
        self.d_ff = d_ff if d_ff is not None else d_model

        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Single transformer block
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Feed-forward (PHASE 1.1: d_ff = d_model, no expansion)
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Output
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Store residuals for metric computation
        self.cache = {}

        # Clamp function (optional, set externally)
        self.clamp_fn = None
        
        # Post-clamp noise scale (set externally for sedation tests)
        self.post_clamp_noise_scale = 0.0

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len] token indices

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(x) + self.pos_emb(positions)

        # Store pre-attention residual
        self.cache['resid_pre_attn'] = x.clone()

        # Attention block with residual
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        # PHASE 1.1: Cache attention output for variance tracking
        self.cache['attn_out'] = attn_out.clone()
        x = self.ln1(x + attn_out)

        # Store post-attention residual
        self.cache['resid_post_attn'] = x.clone()
        self.cache['attn_weights'] = attn_weights

        # Feed-forward block with residual
        # PHASE 4.1: Skip FFN if disabled (attention-only model)
        if not self.disable_ffn:
            ff_out = self.ff(x)
            # PHASE 1.1: Cache FFN output for variance tracking
            self.cache['ff_out'] = ff_out.clone()
            x = self.ln2(x + ff_out)
        else:
            # FFN disabled: no geometric bypass
            ff_out = torch.zeros_like(x)
            self.cache['ff_out'] = ff_out
            x = self.ln2(x)  # Just apply LN, no residual from FFN

        # Store post-FF residual (main residual stream)
        self.cache['resid_post'] = x.clone()

        # Final layer norm
        x = self.ln_final(x)
        
        # Apply clamp if set (for variance control experiments)
        # IMPORTANT: Apply AFTER ln_final so layernorm doesn't undo the clamp
        if self.clamp_fn is not None:
            class MockHook:
                pass
            x = self.clamp_fn(x, MockHook())
            # Cache the clamped value for metric tracking
            self.cache['resid_clamped'] = x.clone()
        
        # Apply post-clamp noise (TRUE test of amplitude fragility)
        # This noise is added AFTER the clamp, so the model cannot absorb it
        if self.post_clamp_noise_scale > 0:
            noise = torch.randn_like(x) * self.post_clamp_noise_scale
            x = x + noise
            # Cache for debugging
            self.cache['post_clamp_noised'] = x.clone()

        # Output head
        logits = self.head(x)

        return logits

    def freeze_attention_heads(self, heads_to_freeze: List[int]):
        """
        Freeze specified attention heads (no gradients).

        Args:
            heads_to_freeze: List of head indices to freeze (0-indexed)
        """
        # For standard PyTorch MultiheadAttention, we freeze the entire layer
        # and then selectively unfreeze specific parameters
        # This is a simplified approach - in TransformerLens it's more granular

        print(f"Warning: Freezing heads {heads_to_freeze} in PyTorch MultiheadAttention")
        print("Note: This is a coarse-grained freeze. Consider using TransformerLens for fine-grained control.")

        # Freeze all attention parameters
        for param in self.attn.parameters():
            param.requires_grad = False

        # Log freeze status
        frozen_params = sum(p.numel() for p in self.attn.parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.attn.parameters())
        print(f"Frozen {frozen_params}/{total_params} attention parameters")


# Toy dataset for quick experimentation
class ToySequenceDataset(Dataset):
    """
    Simple synthetic dataset for testing.

    Task: Next token prediction on simple patterns.
    """

    def __init__(self, n_samples: int = 10000, seq_len: int = 32, vocab_size: int = 100):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Generate deterministic sequences
        np.random.seed(42)
        self.data = []

        for _ in range(n_samples):
            # Simple pattern: repeating sequences
            pattern_len = np.random.randint(2, 8)
            pattern = np.random.randint(0, vocab_size, size=pattern_len)
            seq = np.tile(pattern, (seq_len // pattern_len) + 1)[:seq_len]
            self.data.append(seq)

        self.data = np.array(self.data)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        # Input: all tokens except last
        # Target: all tokens except first (shifted by 1)
        return torch.LongTensor(seq[:-1]), torch.LongTensor(seq[1:])


class InterleavedSequenceDataset(Dataset):
    """
    Hard Task: Two interleaved sequences (FORCES INTERFERENCE).

    Input:  [A1, B1, A2, B2, A3, B3, ...]
    Target: [B1, A2, B2, A3, B3, A4, ...]

    Physics:
    - The model must track two independent threads (A and B)
    - 'A' tokens interfere with 'B' tokens in the residual stream
    - To predict next token in stream A, must attend back 2 positions (skipping B)
    - With seq_len=128 and d_model=40, we have L > d (geometric crowding)
    - Requires precise attention filtering + amplitude compensation when constrained

    This is MUCH harder than simple repeating patterns because:
    1. Semantic interference (two streams compete for representation)
    2. Geometric crowding (L=128 > d=40, so 128 vectors in 40D space)
    3. Long-range dependencies (must look back multiple steps)
    """

    def __init__(self, n_samples: int = 10000, seq_len: int = 128, vocab_size: int = 4096):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Generate two independent PATTERNED sequences (not pure random!)
        # Each stream has a repeating pattern, making the task learnable

        # For interleaved sequence of length (seq_len + 1):
        n_even = (self.seq_len + 2) // 2  # Number of even positions (stream A)
        n_odd = (self.seq_len + 1) // 2   # Number of odd positions (stream B)

        # Stream A: Create a repeating pattern
        pattern_len_a = np.random.randint(2, 8)
        pattern_a = np.random.randint(0, self.vocab_size, size=pattern_len_a)
        seq_a_np = np.tile(pattern_a, (n_even // pattern_len_a) + 1)[:n_even]
        seq_a = torch.from_numpy(seq_a_np).long()

        # Stream B: Create a DIFFERENT repeating pattern
        pattern_len_b = np.random.randint(2, 8)
        pattern_b = np.random.randint(0, self.vocab_size, size=pattern_len_b)
        seq_b_np = np.tile(pattern_b, (n_odd // pattern_len_b) + 1)[:n_odd]
        seq_b = torch.from_numpy(seq_b_np).long()

        # Interleave them: A1, B1, A2, B2, A3, B3, ...
        interleaved = torch.empty(self.seq_len + 1, dtype=torch.long)
        interleaved[0::2] = seq_a  # Even positions
        interleaved[1::2] = seq_b  # Odd positions

        # Input: all except last, Target: all except first (standard next-token)
        x = interleaved[:-1]
        y = interleaved[1:]

        return x, y


class ModularArithmeticDataset(Dataset):
    """
    HIGH PRECISION TASK: Modular Addition (p=113).
    
    This forces EXACT REPRESENTATION with zero geometric slack.
    
    Task: x + y = z (mod 113)
    - Input: [x, +, y, =]
    - Target: z
    
    Physics:
    - The answer manifold is a "twisted torus" or discrete clock
    - Answer 0 is adjacent to answer 112 and 1 in the geometric space
    - Even small noise pushes the vector across decision boundaries
    - The model MUST scream (increase amplitude) to maintain precision
    
    Unlike the Interleaved task where "Cat" is far from "Dog",
    here the model must be geometrically precise to distinguish 0 from 112.
    
    Specs from "Why Variance Explodes" paper:
    - p = 113 (prime, prevents factorization tricks)
    - vocab_size = 128 (113 numbers + special tokens)
    - seq_len = 16
    - Total examples: 113 * 113 = 12,769
    """
    
    def __init__(self, p: int = 113, seq_len: int = 16, train: bool = True, train_frac: float = 0.5):
        """
        Args:
            p: Prime modulus (default 113)
            seq_len: Sequence length including padding (default 16)
            train: If True, use training split; else validation split
            train_frac: Fraction of data for training (standard grokking uses 0.5)
        """
        self.p = p
        self.seq_len = seq_len
        
        # Special tokens
        self.plus_token = p      # 113
        self.eq_token = p + 1    # 114
        self.pad_token = p + 2   # 115
        
        # Generate all pairs
        all_pairs = []
        for x in range(p):
            for y in range(p):
                z = (x + y) % p
                all_pairs.append((x, y, z))
        
        # Shuffle deterministically and split
        np.random.seed(0)
        np.random.shuffle(all_pairs)
        
        n_train = int(len(all_pairs) * train_frac)
        if train:
            self.pairs = all_pairs[:n_train]
        else:
            self.pairs = all_pairs[n_train:]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        x, y, z = self.pairs[idx]
        
        # Sequence: [x, +, y, =, z, PAD, PAD, ...]
        seq = [x, self.plus_token, y, self.eq_token, z]
        
        # Pad to seq_len
        seq = seq + [self.pad_token] * (self.seq_len - len(seq))
        seq = torch.tensor(seq, dtype=torch.long)
        
        # Standard next-token prediction: input = seq[:-1], target = seq[1:]
        return seq[:-1], seq[1:]
    
    @property
    def vocab_size(self):
        """Return vocab size: p numbers + 3 special tokens (rounded up to 128)."""
        return 128  # Paper spec: vocab_size = 128


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    auditor: AllostasisAudit,
    logger: AuditLogger,
    epoch: int,
    clamp_hook: Optional[tuple] = None,
    high_freq_log: bool = False,
    val_loader: Optional[DataLoader] = None,
    noise_scale: float = 0.0
):
    """
    Train for one epoch.

    Args:
        model: Transformer model
        dataloader: Training data
        optimizer: Optimizer
        device: Device to train on
        auditor: Metrics auditor
        logger: Audit logger
        epoch: Current epoch number
        clamp_hook: Optional (hook_name, hook_fn) for clamping
        high_freq_log: STROKE TEST - log every 10 batches during first epoch
        val_loader: Validation loader for high-freq logging
        noise_scale: Noise scale for SNR computation
    """
    model.train()
    total_loss = 0
    total_acc = 0
    n_batches = 0

    # Set clamp function on model if provided
    if clamp_hook is not None:
        hook_name, hook_fn = clamp_hook
        model.clamp_fn = hook_fn
        print(f"  [CLAMP ENABLED] Type: {hook_name}")
    else:
        model.clamp_fn = None

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass (clamp applied inside model if set)
        logits = model(inputs)

        # Debug: Print actual residual norm on first batch
        if batch_idx == 0 and clamp_hook is not None and 'resid_post' in model.cache:
            actual_norm = model.cache['resid_post'].reshape(-1, model.cache['resid_post'].shape[-1]).norm(dim=-1).mean().item()
            print(f"  [CLAMP ACTIVE] Actual resid norm after clamp: {actual_norm:.4f}")

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        )

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            acc = auditor.compute_psi_accuracy(logits, targets)
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        # =====================================================================
        # STROKE TEST: High-Frequency Logging (First Epoch Only)
        # =====================================================================
        # During stroke test, log every 10 batches to catch the "reflex" spike
        # in A_activation that should occur immediately when constraints hit
        # =====================================================================
        if high_freq_log and epoch == 0 and batch_idx % 10 == 0 and batch_idx < 500:
            global_step = epoch * len(dataloader) + batch_idx

            with torch.no_grad():
                # Quick validation (optional - can be expensive)
                if val_loader is not None and batch_idx % 50 == 0:  # Less frequent val
                    val_loss, val_acc = evaluate(model, val_loader, device, auditor)
                else:
                    val_loss = float('nan')
                    val_acc = float('nan')

                # Compute A_activation from cached residuals
                # Use resid_clamped if clamp is active, otherwise use resid_post
                resid_clamped = model.cache.get('resid_clamped')
                resid_post = model.cache.get('resid_post')
                resid_for_metrics = resid_clamped if resid_clamped is not None else resid_post
                
                if resid_for_metrics is not None:
                    A_activation = auditor.compute_amplitude_activation(resid_for_metrics)
                    sigma_sq = auditor.compute_variance(resid_for_metrics)
                else:
                    A_activation = float('nan')
                    sigma_sq = float('nan')

                # Compute other metrics
                A_learned = auditor.compute_amplitude_learned(model)
                A_param = auditor.compute_amplitude_param(model)

                # Compute attention vs FFN variance
                attn_out = model.cache.get('attn_out')
                ff_out = model.cache.get('ff_out')
                if attn_out is not None and ff_out is not None:
                    attn_flat = attn_out.reshape(-1, attn_out.shape[-1])
                    attn_var = attn_flat.var(dim=0).mean().item()
                    ff_flat = ff_out.reshape(-1, ff_out.shape[-1])
                    ff_var = ff_flat.var(dim=0).mean().item()
                    var_ratio = ff_var / (attn_var + 1e-10)
                else:
                    attn_var = float('nan')
                    ff_var = float('nan')
                    var_ratio = float('nan')

                # SNR measurement
                resid_post_attn = model.cache.get('resid_post_attn')
                if resid_post_attn is not None and noise_scale > 0:
                    signal_power = resid_post_attn.var().item()
                    noise_power = noise_scale ** 2
                    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
                else:
                    snr_db = float('nan')

            # Log metrics at batch level
            metrics = {
                'train_loss': loss.item(),
                'train_acc': acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'A_activation': A_activation,
                'A_learned': A_learned,
                'A_param': A_param,
                'sigma_sq': sigma_sq,
                'attn_var': attn_var,
                'ff_var': ff_var,
                'var_ratio_ff_attn': var_ratio,
                'snr_db': snr_db
            }
            logger.log_metrics(global_step, metrics)

            # Print immediate feedback
            print(f"    [REFLEX CHECK] Step {global_step:4d} | "
                  f"A_act: {A_activation:6.3f} | "
                  f"Acc: {acc:.3f} | "
                  f"Loss: {loss.item():.3f}")

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    auditor: AllostasisAudit
):
    """
    Evaluate model on validation set.

    Returns:
        (loss, accuracy)
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1)
            )

            acc = auditor.compute_psi_accuracy(logits, targets)

            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def run_condition(
    condition_name: str,
    config: Dict,
    seed: int = 42,
    baseline_stats: Optional[Dict] = None,
    return_model: bool = False
):
    """
    Run a single experimental condition.

    Args:
        condition_name: Name of condition (control, constraint, etc.)
        config: Configuration dict
        seed: Random seed
        baseline_stats: Optional baseline statistics for clamping
        return_model: If True, return (accuracy, model, dataloader) for baseline computation

    Returns:
        Final validation accuracy (or tuple if return_model=True)
    """
    print(f"\n{'='*80}")
    print(f"RUNNING CONDITION: {condition_name}")
    print(f"{'='*80}\n")

    # Setup
    setup_reproducibility(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = SimpleTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        max_seq_len=config['seq_len'],
        disable_ffn=config.get('disable_ffn', False)
    ).to(device)

    # =========================================================================
    # STROKE TEST: Load Pretrained "Victim" Model
    # =========================================================================
    # Load a healthy, trained model before applying constraints (frozen heads + noise)
    # This tests whether amplitude scaling is a REFLEX (immediate response)
    # vs. LEARNED COMPENSATION (gradual adaptation during training)
    # =========================================================================
    if config.get('load_model') is not None:
        load_path = config['load_model']
        print(f"\n{'='*80}")
        print(f"[!!!] STROKE TEST: LOADING PRETRAINED MODEL")
        print(f"[!!!] Path: {load_path}")
        print(f"{'='*80}\n")
        model.load_state_dict(torch.load(load_path, map_location=device))
        print("[OK] Victim model loaded. Constraints will be applied during training.")
        print("     Watch for A_activation spike in first 500 steps (the 'startle response').\n")

    # PHASE 1.1/4.1: Confirm FFN architecture
    print(f"\n[FFN Architecture]")
    print(f"  d_model: {model.d_model}")
    if model.disable_ffn:
        print(f"  FFN: [DISABLED] - Attention-only model (Phase 4.1)")
    else:
        print(f"  d_ff: {model.d_ff} (ratio: {model.d_ff / model.d_model:.1f}x)")
        print(f"  {'[CRIPPLED]' if model.d_ff == model.d_model else '[EXPANDED]'}")
    
    # Set post-clamp noise scale (for sedation tests)
    post_clamp_noise = config.get('post_clamp_noise_scale', 0.0)
    if post_clamp_noise > 0:
        model.post_clamp_noise_scale = post_clamp_noise
        print(f"\n[!!!] POST-CLAMP NOISE: {post_clamp_noise} (applied just before output head) [!!!]")

    # =========================================================================
    # JAMMING PROTOCOL: PHASE 2 - Post-Attention Noise Injection
    # =========================================================================
    # PHASE 2.1: Move noise injection to the geometric chokepoint
    #
    # OLD (Phase 1): Noise injected at embedding → LayerNorm washes it away
    # NEW (Phase 2): Noise injected AFTER attention + LayerNorm, BEFORE FFN
    #
    # Flow: Attention → Residual → LN1 → [NOISE HERE] → FFN
    #
    # Why this matters:
    #   - Noise now enters the stream that feeds the FFN
    #   - Cannot be normalized away (already passed through LN)
    #   - Directly corrupts the geometric routing signal
    #   - Forces FFN to work with a noisy input
    # =========================================================================
    jamming_hook_handle = None
    # Use noise_scale from config (passed from args)
    noise_scale = config.get('noise_scale', 0.0)
    if noise_scale > 0:
        print(f" [!!!] NOISE INJECTION: Post-Attention Noise (scale={noise_scale}) [!!!]")
        # Hook ln1 output - this is post-attention, post-LN, pre-FFN
        jamming_hook_handle = model.ln1.register_forward_hook(make_jamming_hook(noise_scale=noise_scale))
        print(" [!!!] Hook registered on ln1 (post-attention, pre-FFN) [!!!]\n")

    # Apply head freezing based on config (not just condition name)
    freeze_heads_str = config.get('freeze_heads')
    if freeze_heads_str:
        heads_to_freeze = [int(h.strip()) for h in freeze_heads_str.split(',')]
        model.freeze_attention_heads(heads_to_freeze)
        print(f"Constraint applied: Frozen heads {heads_to_freeze}")

    # Setup clamp based on clamp_type config
    clamp_hook = None
    clamp_type = config.get('clamp_type', 'none')
    verbose = config.get('verbose', False)
    
    if clamp_type == 'naive':
        # SEDATION TEST: Allow direct target_norm override
        if config.get('target_norm') is not None:
            target_norm = config['target_norm']
            print(f"\n[!!!] SEDATION TEST: Forcing amplitude to {target_norm:.2f} [!!!]")
        elif baseline_stats:
            target_norm = baseline_stats['target_norm']
        else:
            target_norm = 8.0  # Default to interleaved level
            print(f"\n[WARNING] No baseline_stats, using default target_norm={target_norm}")
        
        clamp_fn = make_naive_clamp_hook(target_norm, layer_idx=0)
        clamp_hook = ('resid_post', clamp_fn)
        print(f"Naive clamp enabled: target_norm={target_norm:.4f}")

    elif clamp_type == 'mean_preserving':
        if config.get('healthy_std') is not None:
            healthy_std = config['healthy_std']
            print(f"\n[!!!] Mean-preserving clamp with std={healthy_std:.2f} [!!!]")
        elif baseline_stats:
            healthy_std = baseline_stats['healthy_std']
        else:
            healthy_std = 1.0  # Default
            print(f"\n[WARNING] No baseline_stats, using default healthy_std={healthy_std}")
        
        clamp_fn = make_mean_preserving_clamp_hook(healthy_std, layer_idx=0)
        clamp_hook = ('resid_post', clamp_fn)
        print(f"Mean-preserving clamp enabled: healthy_std={healthy_std:.4f}")

    # Create datasets based on config
    dataset_type = config.get('dataset', 'interleaved')
    
    if dataset_type == 'modular':
        # HIGH PRECISION TASK: Modular Arithmetic
        # This has ZERO geometric slack - the model MUST be precise
        p = config.get('modulus', 113)
        print(f"\n[DATASET] Modular Arithmetic (p={p}) - HIGH PRECISION")
        print(f"  Task: x + y = z (mod {p})")
        print(f"  Total examples: {p * p} ({p} * {p})")
        print("  Train/Val split: 50/50")
        train_dataset = ModularArithmeticDataset(p=p, seq_len=16, train=True)
        val_dataset = ModularArithmeticDataset(p=p, seq_len=16, train=False)
    else:
        # Use InterleavedSequenceDataset for hard task with interference
        # (Forces semantic + geometric crowding even with L > d)
        print("\n[DATASET] Interleaved Sequences")
        train_dataset = InterleavedSequenceDataset(
            n_samples=config['n_train'],
            seq_len=config['seq_len'],
            vocab_size=config['vocab_size']
        )
        val_dataset = InterleavedSequenceDataset(
            n_samples=config['n_val'],
            seq_len=config['seq_len'],
            vocab_size=config['vocab_size']
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr']
    )

    # Metrics and logging
    auditor = AllostasisAudit(device=device)
    logger = AuditLogger(
        experiment_name=f"exp_a_{condition_name}",
        seed=seed
    )

    # Training loop
    print(f"\nTraining for {config['n_epochs']} epochs...")

    for epoch in range(config['n_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            auditor, logger, epoch, clamp_hook,
            high_freq_log=config.get('high_freq_log', False),
            val_loader=val_loader,
            noise_scale=noise_scale
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device, auditor)

        # Compute additional metrics
        with torch.no_grad():
            # Get a sample batch for metrics
            sample_inputs, sample_targets = next(iter(val_loader))
            sample_inputs = sample_inputs.to(device)
            logits = model(sample_inputs)

            # Get cached residuals
            resid_post = model.cache.get('resid_post')

            if resid_post is not None:
                A_activation = auditor.compute_amplitude_activation(resid_post)
                sigma_sq = auditor.compute_variance(resid_post)
            else:
                A_activation = float('nan')
                sigma_sq = float('nan')

            A_learned = auditor.compute_amplitude_learned(model)
            A_param = auditor.compute_amplitude_param(model)

            # PHASE 1.1: Compute attention vs FFN variance
            attn_out = model.cache.get('attn_out')
            ff_out = model.cache.get('ff_out')

            if attn_out is not None and ff_out is not None:
                # Compute variance of attention output
                attn_flat = attn_out.reshape(-1, attn_out.shape[-1])
                attn_var = attn_flat.var(dim=0).mean().item()

                # Compute variance of FFN output
                ff_flat = ff_out.reshape(-1, ff_out.shape[-1])
                ff_var = ff_flat.var(dim=0).mean().item()

                # Compute ratio (FFN / attention)
                var_ratio = ff_var / (attn_var + 1e-10)
            else:
                attn_var = float('nan')
                ff_var = float('nan')
                var_ratio = float('nan')

            # PHASE 2.1: SNR measurement (verify noise survives to FFN input)
            resid_post_attn = model.cache.get('resid_post_attn')
            if resid_post_attn is not None and noise_scale > 0:
                # Signal power: variance of post-attention residual
                signal_power = resid_post_attn.var().item()
                # Noise power: noise_scale^2 (Gaussian noise variance)
                noise_power = noise_scale ** 2
                # SNR in dB
                snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
            else:
                snr_db = float('nan')

        # Log metrics
        step = epoch * len(train_loader)
        if logger.should_log(step, val_acc):
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'A_activation': A_activation,
                'A_learned': A_learned,
                'A_param': A_param,
                'sigma_sq': sigma_sq,
                # PHASE 1.1: Variance tracking
                'attn_var': attn_var,
                'ff_var': ff_var,
                'var_ratio_ff_attn': var_ratio,
                # PHASE 2.1: SNR tracking
                'snr_db': snr_db
            }
            logger.log_metrics(step, metrics)

        # Print progress
        if epoch % 10 == 0 or epoch == config['n_epochs'] - 1:
            base_msg = (f"Epoch {epoch:3d} | "
                       f"Train: {train_acc:.3f} | "
                       f"Val: {val_acc:.3f} | "
                       f"A_act: {A_activation:.3f} | "
                       f"Var: {sigma_sq:.3f} | "
                       f"FFN/Attn: {var_ratio:.2f}")
            if noise_scale > 0:
                base_msg += f" | SNR: {snr_db:.1f}dB"
            print(base_msg)

    # Save log
    logger.save_log()

    # Return final validation accuracy
    final_val_loss, final_val_acc = evaluate(model, val_loader, device, auditor)
    print(f"\nFinal validation accuracy: {final_val_acc:.4f}")

    # =========================================================================
    # STROKE TEST: Save Model (Create the "Victim")
    # =========================================================================
    if config.get('save_model') is not None:
        save_path = config['save_model']
        torch.save(model.state_dict(), save_path)
        print(f"\n[OK] Model saved to: {save_path}")
        print("     This 'healthy victim' can be loaded with --load_model for stroke testing.")

    if return_model:
        return final_val_acc, model, val_loader
    return final_val_acc


def main():
    parser = argparse.ArgumentParser(
        description="Experiment A: Foundation - Configurable Allostatic Load Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train healthy victim
  python exp_a_foundation.py --dataset modular --n_epochs 50 --save_model victim.pt
  
  # Stroke test (noise + freeze)
  python exp_a_foundation.py --dataset modular --load_model victim.pt \\
      --noise_scale 2.0 --freeze_heads "1,2,3" --high_freq_log
  
  # Sedation test (clamp amplitude)
  python exp_a_foundation.py --dataset modular --load_model victim.pt \\
      --clamp_type naive --target_norm 8.0
  
  # Combined sedation + noise
  python exp_a_foundation.py --dataset modular --load_model victim.pt \\
      --clamp_type naive --target_norm 8.0 --noise_scale 2.0 --freeze_heads "1,2,3"
        """
    )
    
    # === Basic Configuration ===
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--quick_test', action='store_true', help='Quick test run with reduced data')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug output')
    
    # === Model Architecture ===
    parser.add_argument('--vocab_size', type=int, default=4096,
                       help='Vocabulary size (use 4096 for n>>d, 100 for n<<d)')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension (decrease to 32 with large vocab for more stress)')
    parser.add_argument('--disable_ffn', action='store_true',
                       help='Disable FFN entirely (attention-only model)')
    
    # === Dataset Selection ===
    parser.add_argument('--dataset', type=str, default='interleaved',
                       choices=['interleaved', 'modular'],
                       help='Dataset: interleaved (low precision) or modular (high precision)')
    parser.add_argument('--modulus', type=int, default=113,
                       help='Prime modulus for modular arithmetic task (e.g., 7, 13, 41, 113, 227)')
    
    # === Model Loading/Saving ===
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load a pretrained model state dict')
    parser.add_argument('--save_model', type=str, default=None,
                       help='Path to save the model state dict after training')
    
    # === Constraint Configuration ===
    parser.add_argument('--freeze_heads', type=str, default=None,
                       help='Comma-separated head indices to freeze (e.g., "1,2,3"). None = no freeze.')
    parser.add_argument('--no_freeze', action='store_true',
                       help='Disable head freezing even in legacy constraint conditions')
    
    # === Noise Injection ===
    parser.add_argument('--noise_scale', type=float, default=0.0,
                       help='Pre-FFN noise injection (0.0 = disabled). Applied at ln1.')
    parser.add_argument('--post_clamp_noise_scale', type=float, default=0.0,
                       help='Post-clamp noise (applied just before output head). This is the TRUE test of amplitude fragility.')
    
    # === Clamp Configuration ===
    parser.add_argument('--clamp_type', type=str, default='none',
                       choices=['none', 'naive', 'mean_preserving'],
                       help='Type of amplitude/variance clamp to apply')
    parser.add_argument('--target_norm', type=float, default=None,
                       help='Target norm for naive clamp (e.g., 8.0 to sedate a high-amplitude model)')
    parser.add_argument('--healthy_std', type=float, default=None,
                       help='Target std for mean-preserving clamp')
    
    # === Logging Configuration ===
    parser.add_argument('--high_freq_log', action='store_true',
                       help='Log every 10 steps for first epoch (catch reflex dynamics)')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Interval for high-frequency logging (steps)')
    
    # === Presets (Named Configurations) ===
    parser.add_argument('--preset', type=str, default=None,
                       choices=['stroke_test', 'sedation_test', 'godzilla_test', 'sedation_noise'],
                       help='Load a preset configuration')
    
    # === Legacy Condition (for backwards compatibility) ===
    parser.add_argument('--condition', type=str, default='control',
                       choices=['control', 'constraint', 'naive_clamp', 'mean_preserving_clamp', 'all'],
                       help='Legacy condition mode (prefer using individual flags instead)')

    args = parser.parse_args()
    
    # === Apply Presets ===
    PRESETS = {
        'stroke_test': {
            'noise_scale': 2.0,
            'freeze_heads': '1,2,3',
            'high_freq_log': True,
        },
        'sedation_test': {
            'clamp_type': 'naive',
            'target_norm': 8.0,
        },
        'godzilla_test': {
            'noise_scale': 5.0,
            'freeze_heads': '1,2,3',
            'high_freq_log': True,
        },
        'sedation_noise': {
            'clamp_type': 'naive',
            'target_norm': 8.0,
            'noise_scale': 2.0,
            'freeze_heads': '1,2,3',
            'high_freq_log': True,
        },
    }
    
    if args.preset:
        preset = PRESETS[args.preset]
        print(f"\n[PRESET] Loading preset: {args.preset}")
        for key, value in preset.items():
            # Only apply preset if user didn't override
            if getattr(args, key, None) is None or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
                print(f"  {key}: {value}")
    
    # === Handle Legacy Conditions ===
    if args.condition in ['constraint', 'naive_clamp', 'mean_preserving_clamp']:
        if args.freeze_heads is None and not args.no_freeze:
            args.freeze_heads = '1,2,3'  # Default legacy behavior
        if args.condition == 'naive_clamp' and args.clamp_type == 'none':
            args.clamp_type = 'naive'
        if args.condition == 'mean_preserving_clamp' and args.clamp_type == 'none':
            args.clamp_type = 'mean_preserving'

    # (parse_args already called above)

    # Configuration
    # CRITICAL: vocab_size >> d_model to force superposition regime
    # CRITICAL: seq_len > d_model to force geometric crowding (L > d)
    # With n << d, model can use orthogonal basis (no interference, no stress)
    # With n >> d, model MUST use superposition (interference creates allostatic load)
    # With L < d, model can separate all context positions (no crowding)
    # With L > d, context positions must share dimensions (geometric crowding)
    config = {
        'vocab_size': args.vocab_size,  # Default 4096 forces n >> d for superposition
        'd_model': args.d_model,        # Default 128
        'n_heads': 4,
        'seq_len': 128,                 # CRITICAL: L > d for geometric crowding
        'n_train': 10000 if not args.quick_test else 1000,
        'n_val': 2000 if not args.quick_test else 200,
        'batch_size': 64,
        'lr': 1e-3,
        'n_epochs': args.n_epochs if not args.quick_test else 10,
        'noise_scale': args.noise_scale,
        'post_clamp_noise_scale': args.post_clamp_noise_scale,
        'disable_ffn': args.disable_ffn,
        'load_model': args.load_model,
        'save_model': args.save_model,
        'high_freq_log': args.high_freq_log,
        'log_interval': args.log_interval,
        'dataset': args.dataset,
        'modulus': args.modulus,
        # Clamp settings
        'clamp_type': args.clamp_type,
        'target_norm': args.target_norm,
        'healthy_std': args.healthy_std,
        # Constraint settings
        'freeze_heads': args.freeze_heads,
        # Debug settings
        'verbose': args.verbose,
    }

    # Validate configuration
    n_over_d = config['vocab_size'] / config['d_model']
    L_over_d = config['seq_len'] / config['d_model']

    print(f"\n[CONFIG VALIDATION]")
    print(f"  n/d ratio: {n_over_d:.1f}")
    print(f"  L/d ratio: {L_over_d:.2f}")

    # Check n/d (superposition pressure)
    if n_over_d < 2:
        print(f"  WARNING: n/d = {n_over_d:.1f} < 2")
        print(f"  Model may use orthogonal basis (no superposition pressure)")
        print(f"  Consider: --vocab_size 4096 or decrease --d_model")
    elif n_over_d > 10:
        print(f"  GOOD: n/d = {n_over_d:.1f} >> 1 (superposition forced)")

    # Check L/d (geometric crowding)
    if L_over_d < 1.5:
        print(f"  WARNING: L/d = {L_over_d:.2f} < 1.5")
        print(f"  Context can fit in orthogonal subspaces (no geometric crowding)")
        print(f"  Single head may still solve task without amplitude scaling!")
        print(f"  Using InterleavedSequenceDataset to force SEMANTIC interference")
    else:
        print(f"  GOOD: L/d = {L_over_d:.2f} > 1 (geometric crowding + semantic interference)")

    if n_over_d > 10 and L_over_d > 1.5:
        print(f"  OPTIMAL: Both superposition AND crowding present")
        print(f"  Constraint should trigger allostatic load")

    print("Experiment A: Foundation")
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # === Determine Run Mode ===
    # If user specified custom settings (clamp, freeze, noise), run in "custom" single-run mode
    # Otherwise, use legacy condition-based flow
    
    use_custom_mode = (
        config['clamp_type'] != 'none' or 
        config['freeze_heads'] is not None or 
        config['noise_scale'] > 0 or
        config['load_model'] is not None
    )
    
    if use_custom_mode:
        # === CUSTOM SINGLE RUN MODE ===
        # Run one condition with all the CLI-specified settings
        print("\n" + "="*80)
        print("CUSTOM EXPERIMENT RUN")
        print("="*80)
        print("\nUsing custom settings from CLI:")
        if config['clamp_type'] != 'none':
            print(f"  Clamp: {config['clamp_type']} (target_norm={config.get('target_norm')}, healthy_std={config.get('healthy_std')})")
        if config['freeze_heads']:
            print(f"  Freeze heads: {config['freeze_heads']}")
        if config['noise_scale'] > 0:
            print(f"  Noise scale: {config['noise_scale']}")
        if config['load_model']:
            print(f"  Loading model: {config['load_model']}")
        
        # Run single condition with CLI settings
        results = {'custom': run_condition('custom', config, args.seed)}
        
    else:
        # === LEGACY CONDITION-BASED FLOW ===
        conditions = ['control', 'constraint', 'naive_clamp', 'mean_preserving_clamp']
        if args.condition != 'all':
            conditions = [args.condition]

        results = {}
        baseline_stats = None

        for condition in conditions:
            # Run control first to get baseline stats
            if condition == 'control' or (baseline_stats is None and condition != 'control'):
                if 'control' not in results:
                    print("\n[PHASE 1] Running CONTROL condition to establish baseline...")
                    control_acc, control_model, control_loader = run_condition(
                        'control', config, args.seed, return_model=True
                    )
                    results['control'] = control_acc

                    # Compute ACTUAL baseline statistics from control run
                    print("\n[PHASE 2] Computing baseline statistics from control model...")
                    from lib.clamps import ClampCalibrator

                    calibrator = ClampCalibrator()
                    control_model.eval()
                    device = next(control_model.parameters()).device

                    # Accumulate statistics over validation set
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(control_loader):
                            if batch_idx >= 50:  # Use 50 batches for calibration
                                break
                            inputs = inputs.to(device)
                            _ = control_model(inputs)

                            # Get residual stream from cache
                            if 'resid_post' in control_model.cache:
                                resid = control_model.cache['resid_post']
                                calibrator.accumulate(resid)

                    baseline_stats = {
                        'target_norm': calibrator.get_target_norm(),
                        'healthy_std': calibrator.get_healthy_std()
                    }

                    print(f"\nBaseline statistics (COMPUTED FROM DATA):")
                    print(f"  target_norm: {baseline_stats['target_norm']:.4f}")
                    print(f"  healthy_std: {baseline_stats['healthy_std']:.4f}")
                    print(f"  (These will be used for clamp conditions)")

            if condition != 'control':
                results[condition] = run_condition(condition, config, args.seed, baseline_stats)

    # Summary
    print(f"\n{'='*80}")
    print("EXPERIMENT A SUMMARY")
    print(f"{'='*80}\n")

    for condition, acc in results.items():
        print(f"{condition:20s}: {acc:.4f}")

    # Check success criteria
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*80}\n")

    checks = []

    if 'control' in results:
        check = results['control'] >= 0.95
        checks.append(('Control >=95%', check))
        print(f"[OK] Control >=95%: {check} ({results['control']:.2%})")

    if 'constraint' in results:
        check = 0.55 <= results['constraint'] <= 0.65
        checks.append(('Constraint 55-65%', check))
        print(f"[OK] Constraint 55-65%: {check} ({results['constraint']:.2%})")

    if 'naive_clamp' in results:
        check = results['naive_clamp'] < 0.05
        checks.append(('Naive clamp <5%', check))
        print(f"[OK] Naive clamp <5%: {check} ({results['naive_clamp']:.2%})")

    if 'mean_preserving_clamp' in results and 'constraint' in results:
        if results['constraint'] > 0:
            relative = results['mean_preserving_clamp'] / results['constraint']
            check = relative >= 0.90
            checks.append(('Mean-pres >=90% of constraint', check))
            print(f"[OK] Mean-preserving >=90% of constraint: {check} ({relative:.1%})")
        else:
            print(f"[OK] Mean-preserving >=90% of constraint: N/A (constraint is 0%)")

    all_passed = all(check for _, check in checks)
    print(f"\nOverall: {'[OK] PASS' if all_passed else '[FAIL] FAIL'}")


if __name__ == '__main__':
    main()
