"""
Phase 6 Experiment O.0: Baseline Othello-GPT Training (OPTIMIZED)

Optimizations over original:
1. Fixed resume bug (training loop respects start_step)
2. Mixed precision training (AMP) for 2x speed on L4
3. Best model checkpointing
4. Complete checkpoint saves (includes architecture, optimizer, step)
5. Reduced memory caching
6. Gradient accumulation support
7. Pre-generates cache if missing
8. More frequent early checkpoints

Target: ~99.9% legal move accuracy
Hardware: NVIDIA L4 (24GB VRAM)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, Tuple, Optional
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import gc

# Enable TF32 for faster matmul on Ampere+
torch.set_float32_matmul_precision('high')

from lib.logging_utils import setup_reproducibility
from lib.deep_transformer import DeepTransformer
from lib.othello_dataset import (
    OthelloDataset, get_othello_loaders,
    VOCAB_SIZE, PASS_TOKEN
)


# Architecture spec from protocol
OTHELLO_ARCH = {
    'n_layers': 4,
    'd_model': 512,
    'n_heads': 8,
    'd_ff': 2048,
    'vocab_size': VOCAB_SIZE,  # 65
    'ctx_len': 60,
    'dropout': 0.0,
}


def validate(model: nn.Module, dataloader: DataLoader, device, use_amp: bool = True) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_legal_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            moves, legal_targets, next_move = batch
            moves = moves.to(device, non_blocking=True)
            legal_targets = legal_targets.to(device, non_blocking=True)
            next_move = next_move.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                logits = model(moves)
                last_logits = logits[:, -1, :]
                loss = F.cross_entropy(last_logits, next_move)

            preds = last_logits.argmax(dim=-1)
            is_legal = legal_targets[torch.arange(preds.shape[0], device=device), preds]

            total_loss += loss.item() * moves.shape[0]
            total_legal_correct += is_legal.sum().item()
            total_samples += moves.shape[0]

    return {
        'legal_acc': total_legal_correct / total_samples,
        'loss': total_loss / total_samples
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    step: int,
    best_acc: float,
    trajectory: list,
    architecture: dict,
    seed: int,
    is_best: bool = False
):
    """Save a complete checkpoint that can be fully resumed."""
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_acc': best_acc,
        'trajectory': trajectory,
        'architecture': architecture,
        'seed': seed,
        'is_best': is_best,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, path)
    return path


def train_baseline(
    n_train_games: int = 100000,
    n_val_games: int = 10000,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    training_steps: int = 50000,
    log_interval: int = 500,
    save_interval: int = 5000,
    seed: int = 42,
    device: str = None,
    resume_path: str = None,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    save_best: bool = True,
    warmup_steps: int = 0,
    resume_fresh_optimizer: bool = False
):
    """
    Train baseline Othello-GPT model with optimizations.

    Args:
        use_amp: Use automatic mixed precision (recommended for L4)
        grad_accum_steps: Gradient accumulation steps (for larger effective batch)
        save_best: Save checkpoint whenever best_acc improves
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    setup_reproducibility(seed)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("="*60)
    print("PHASE 6 O.0: BASELINE OTHELLO-GPT TRAINING (OPTIMIZED)")
    print("="*60)
    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Mixed Precision: {use_amp}")
    print(f"Gradient Accumulation: {grad_accum_steps}")
    print(f"Effective Batch Size: {batch_size * grad_accum_steps}")
    print(f"Architecture: {OTHELLO_ARCH}")
    if resume_path:
        print(f"Resuming from: {resume_path}")
    print()

    # Load data with precomputation
    print("Loading Othello games...")
    print("  (This may take a while if cache needs to be generated)")
    train_loader, val_loader = get_othello_loaders(
        n_train_games=n_train_games,
        n_val_games=n_val_games,
        batch_size=batch_size,
        ctx_len=OTHELLO_ARCH['ctx_len'],
        seed=seed
    )
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Val samples: {len(val_loader.dataset):,}")
    print()

    # Create model
    model = DeepTransformer(
        vocab_size=OTHELLO_ARCH['vocab_size'],
        d_model=OTHELLO_ARCH['d_model'],
        n_heads=OTHELLO_ARCH['n_heads'],
        n_layers=OTHELLO_ARCH['n_layers'],
        d_ff=OTHELLO_ARCH['d_ff'],
        max_seq_len=OTHELLO_ARCH['ctx_len'],
        dropout=OTHELLO_ARCH['dropout']
    )
    model.to(device)

    # Disable activation caching during training (saves memory)
    model.cache_activations = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=use_amp)

    # Learning rate scheduler with warmup
    def get_lr(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps  # Linear warmup
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # Resume from checkpoint if provided
    start_step = 0
    trajectory = []
    best_acc = 0.0

    if resume_path:
        print(f"Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Check what's available in checkpoint
        has_optimizer = 'optimizer_state_dict' in checkpoint
        has_step = 'step' in checkpoint

        if not has_optimizer:
            print("  WARNING: No optimizer state in checkpoint!")
            print("  Using fresh optimizer - recommend using --warmup_steps 2000")
            resume_fresh_optimizer = True

        if not has_step:
            print("  WARNING: No step count in checkpoint!")
            print("  Starting from step 0 with current model weights")

        # Load optimizer state unless using fresh
        if has_optimizer and not resume_fresh_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  Loaded optimizer state")
        else:
            print("  Using fresh optimizer" + (f" with {warmup_steps} warmup steps" if warmup_steps > 0 else ""))

        if 'scaler_state_dict' in checkpoint and not resume_fresh_optimizer:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if has_step and not resume_fresh_optimizer:
            start_step = checkpoint['step']
        if 'trajectory' in checkpoint:
            trajectory = checkpoint['trajectory']
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']

        # If resuming with fresh optimizer, evaluate current model accuracy
        if resume_fresh_optimizer or not has_optimizer:
            print("  Evaluating current model accuracy...")
            current_metrics = validate(model, val_loader, device, use_amp)
            best_acc = max(best_acc, current_metrics['legal_acc'])
            print(f"  Current model accuracy: {current_metrics['legal_acc']*100:.2f}%")
            print(f"  Best accuracy to beat: {best_acc*100:.2f}%")

        print(f"  Resume config: start_step={start_step}, best_acc={best_acc*100:.2f}%")

        # Clear CUDA cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # Calculate remaining steps
    remaining_steps = training_steps - start_step
    if remaining_steps <= 0:
        print(f"Training already complete (step {start_step} >= {training_steps})")
        return model, trajectory

    print(f"\nTraining from step {start_step} to {training_steps} ({remaining_steps} steps)")
    print()

    # Training loop
    train_iter = iter(train_loader)
    optimizer.zero_grad()

    # FIXED: Loop starts from start_step, not 0
    pbar = tqdm(range(start_step, training_steps), desc="Training", initial=start_step, total=training_steps)

    for step in pbar:
        model.train()

        # Gradient accumulation loop
        accum_loss = 0.0
        for accum_step in range(grad_accum_steps):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            moves, legal_targets, next_move = batch
            moves = moves.to(device, non_blocking=True)
            next_move = next_move.to(device, non_blocking=True)

            # Forward with AMP
            with autocast(enabled=use_amp):
                logits = model(moves)
                last_logits = logits[:, -1, :]
                loss = F.cross_entropy(last_logits, next_move)
                loss = loss / grad_accum_steps  # Scale for accumulation

            # Backward with scaler
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # LR scheduler step (for warmup)
        scheduler.step()

        # Logging
        current_step = step + 1
        if current_step % log_interval == 0:
            metrics = validate(model, val_loader, device, use_amp)
            trajectory.append({
                'step': current_step,
                'train_loss': accum_loss * grad_accum_steps,  # Unscale
                **metrics
            })

            pbar.set_postfix({
                'loss': f"{accum_loss * grad_accum_steps:.4f}",
                'val_acc': f"{metrics['legal_acc']*100:.2f}%",
                'best': f"{best_acc*100:.1f}%"
            })

            # Save best model
            if metrics['legal_acc'] > best_acc:
                best_acc = metrics['legal_acc']
                if save_best:
                    best_path = Path(f"data/othello_baseline_{run_id}_best.pt")
                    save_checkpoint(
                        best_path, model, optimizer, scaler,
                        current_step, best_acc, trajectory,
                        OTHELLO_ARCH, seed, is_best=True
                    )
                    tqdm.write(f"  New best: {best_acc*100:.2f}% saved to {best_path}")

        # Periodic checkpointing
        if current_step % save_interval == 0:
            checkpoint_path = Path(f"data/othello_baseline_{run_id}_step{current_step}.pt")
            save_checkpoint(
                checkpoint_path, model, optimizer, scaler,
                current_step, best_acc, trajectory,
                OTHELLO_ARCH, seed
            )
            tqdm.write(f"  Checkpoint saved: {checkpoint_path}")

    # Final evaluation
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    final_metrics = validate(model, val_loader, device, use_amp)
    print(f"Final Legal Move Accuracy: {final_metrics['legal_acc']*100:.2f}%")
    print(f"Best Legal Move Accuracy: {best_acc*100:.2f}%")

    # Save final model (complete checkpoint)
    final_path = Path(f"data/othello_baseline_{run_id}_final.pt")
    save_checkpoint(
        final_path, model, optimizer, scaler,
        training_steps, best_acc, trajectory,
        OTHELLO_ARCH, seed
    )
    final_path_2 = final_path.with_name(final_path.stem + "_final_metrics.json")
    with open(final_path_2, 'w') as f:
        json.dump({
            'final_metrics': final_metrics,
            'best_acc': best_acc,
            'total_steps': training_steps,
            'trajectory_length': len(trajectory)
        }, f, indent=2)

    print(f"\nFinal model saved: {final_path}")
    print(f"Metrics saved: {final_path_2}")

    return model, trajectory


def main():
    parser = argparse.ArgumentParser(description='Train baseline Othello-GPT (Optimized)')
    parser.add_argument('--n_train_games', type=int, default=100000)
    parser.add_argument('--n_val_games', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--training_steps', type=int, default=100000,
                        help='Total training steps (default: 100k for ~99% accuracy)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick_test', action='store_true', help='Quick test with minimal data')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='LR warmup steps (use ~1000-2000 when resuming without optimizer state)')
    parser.add_argument('--resume_fresh_optimizer', action='store_true',
                        help='When resuming, ignore saved optimizer and use fresh one with warmup')

    args = parser.parse_args()

    if args.quick_test:
        args.n_train_games = 1000
        args.n_val_games = 200
        args.training_steps = 1000
        args.log_interval = 100
        args.save_interval = 500
        print("\n*** QUICK TEST MODE ***\n")

    train_baseline(
        n_train_games=args.n_train_games,
        n_val_games=args.n_val_games,
        batch_size=args.batch_size,
        lr=args.lr,
        training_steps=args.training_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        resume_path=args.resume,
        use_amp=not args.no_amp,
        grad_accum_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        resume_fresh_optimizer=args.resume_fresh_optimizer
    )


if __name__ == '__main__':
    main()
