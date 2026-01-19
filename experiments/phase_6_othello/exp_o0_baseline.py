"""
Phase 6 Experiment O.0: Baseline Othello-GPT Training

Train a transformer model to predict legal moves in Othello.
This establishes the baseline for all subsequent G×S experiments.

Target: ~99.9% legal move accuracy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple
import argparse
import json
from datetime import datetime
from tqdm import tqdm

# Enable TF32 for A100 (up to 8x faster)
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


def compute_legal_move_accuracy(logits: torch.Tensor, legal_targets: torch.Tensor) -> float:
    """
    Compute accuracy: Is the top-1 prediction a legal move?
    
    Args:
        logits: [batch, seq_len, vocab_size] or [batch, vocab_size]
        legal_targets: [batch, vocab_size] binary mask
        
    Returns:
        Fraction of predictions that are legal
    """
    # Take last position if sequence
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # [batch, vocab]
    
    # Get top prediction
    preds = logits.argmax(dim=-1)  # [batch]
    
    # Check if prediction is legal
    batch_size = preds.shape[0]
    is_legal = legal_targets[torch.arange(batch_size), preds]  # [batch]
    
    return is_legal.float().mean().item()


def validate(model: nn.Module, dataloader: DataLoader, device) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    
    Returns:
        Dict with 'legal_acc' and 'loss'
    """
    model.eval()
    total_loss = 0.0
    total_legal_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            moves, legal_targets, next_move = batch
            moves = moves.to(device)
            legal_targets = legal_targets.to(device)
            next_move = next_move.to(device)
            
            logits = model(moves)
            
            # Loss on next move prediction
            last_logits = logits[:, -1, :]
            loss = F.cross_entropy(last_logits, next_move)
            
            # Legal move accuracy
            preds = last_logits.argmax(dim=-1)
            is_legal = legal_targets[torch.arange(preds.shape[0], device=device), preds]
            
            total_loss += loss.item() * moves.shape[0]
            total_legal_correct += is_legal.sum().item()
            total_samples += moves.shape[0]
    
    return {
        'legal_acc': total_legal_correct / total_samples,
        'loss': total_loss / total_samples
    }


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
    resume_path: str = None
):
    """
    Train baseline Othello-GPT model.
    
    Args:
        resume_path: Path to checkpoint to resume from
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    setup_reproducibility(seed)
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*60)
    print("PHASE 6 O.0: BASELINE OTHELLO-GPT TRAINING")
    print("="*60)
    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    print(f"Architecture: {OTHELLO_ARCH}")
    if resume_path:
        print(f"Resuming from: {resume_path}")
    print()
    
    # Load data
    print("Loading Othello games...")
    train_loader, val_loader = get_othello_loaders(
        n_train_games=n_train_games,
        n_val_games=n_val_games,
        batch_size=batch_size,
        ctx_len=OTHELLO_ARCH['ctx_len'],
        seed=seed
    )
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
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Resume from checkpoint if provided
    start_step = 0
    trajectory = []
    best_acc = 0.0
    
    if resume_path:
        print(f"Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Check for optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  Loaded optimizer state")
        else:
            print("  WARNING: No optimizer state in checkpoint!")
            print("  Resuming with fresh optimizer may cause degradation.")
            print("  Consider using exp_o0_baseline_optimized.py with --warmup_steps")

        if 'step' in checkpoint:
            start_step = checkpoint['step']
        else:
            print("  WARNING: No step count in checkpoint, starting from step 0")

        if 'trajectory' in checkpoint:
            trajectory = checkpoint['trajectory']
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        print(f"  Resumed from step {start_step}, best_acc={best_acc*100:.2f}%")
    
    # Training loop
    train_iter = iter(train_loader)

    # FIXED: Loop starts from start_step, not 0
    pbar = tqdm(range(start_step, training_steps), desc="Training", initial=start_step, total=training_steps)
    for step in pbar:
        model.train()
        
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        moves, legal_targets, next_move = batch
        moves = moves.to(device)
        legal_targets = legal_targets.to(device)
        next_move = next_move.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(moves)
        
        # CE loss on next move
        last_logits = logits[:, -1, :]
        loss = F.cross_entropy(last_logits, next_move)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if (step + 1) % log_interval == 0:
            metrics = validate(model, val_loader, device)
            trajectory.append({
                'step': step + 1,
                'train_loss': loss.item(),
                **metrics
            })
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'val_acc': f"{metrics['legal_acc']*100:.2f}%"
            })
            
            if metrics['legal_acc'] > best_acc:
                best_acc = metrics['legal_acc']
        
        # Checkpointing
        if (step + 1) % save_interval == 0:
            checkpoint_path = Path(f"data/othello_baseline_{run_id}_step{step+1}.pt")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'trajectory': trajectory
            }, checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    final_metrics = validate(model, val_loader, device)
    print(f"Final Legal Move Accuracy: {final_metrics['legal_acc']*100:.2f}%")
    print(f"Best Legal Move Accuracy: {best_acc*100:.2f}%")
    
    # Save final model (FIXED: include optimizer state and step for proper resume)
    final_path = Path(f"data/othello_baseline_{run_id}_final.pt")
    torch.save({
        'step': training_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'architecture': OTHELLO_ARCH,
        'final_metrics': final_metrics,
        'trajectory': trajectory,
        'seed': seed
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    
    return model, trajectory


def main():
    parser = argparse.ArgumentParser(description='Train baseline Othello-GPT')
    parser.add_argument('--n_train_games', type=int, default=100000)
    parser.add_argument('--n_val_games', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--training_steps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick_test', action='store_true', help='Quick test with minimal data')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.n_train_games = 1000
        args.n_val_games = 200
        args.training_steps = 500
        print("\n*** QUICK TEST MODE ***\n")
    
    train_baseline(
        n_train_games=args.n_train_games,
        n_val_games=args.n_val_games,
        batch_size=args.batch_size,
        lr=args.lr,
        training_steps=args.training_steps,
        seed=args.seed,
        resume_path=args.resume
    )


if __name__ == '__main__':
    main()
