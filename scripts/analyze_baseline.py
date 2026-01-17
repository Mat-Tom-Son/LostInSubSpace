"""Analyze Phase 6 baseline model."""
import torch
import sys
sys.path.insert(0, 'clean_audit')
from lib.deep_transformer import DeepTransformer
from lib.othello_dataset import OthelloDataset, VOCAB_SIZE, get_othello_loaders
import numpy as np

# Load model
ckpt_path = 'clean_audit/data/othello_games/LostInSubSpace_clean_audit_clean_audit_data_othello_baseline_20260116_195406_final.pt'
d = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Get architecture
arch = d.get('architecture', {})
print('=== MODEL ARCHITECTURE ===')
print(f"Layers: {arch.get('n_layers', 4)}")
print(f"d_model: {arch.get('d_model', 512)}")
print(f"Heads: {arch.get('n_heads', 8)}")
print(f"Vocab: {arch.get('vocab_size', 65)}")
print(f"Final Accuracy: {d['final_metrics']['legal_acc']*100:.2f}%")

# Load model
model = DeepTransformer(
    vocab_size=arch.get('vocab_size', 65),
    d_model=arch.get('d_model', 512),
    n_heads=arch.get('n_heads', 8),
    n_layers=arch.get('n_layers', 4),
    max_seq_len=arch.get('ctx_len', 60)
)
model.load_state_dict(d['model_state_dict'])
model.eval()
print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Get some test data
print("\n=== LOADING TEST DATA ===")
train_loader, val_loader = get_othello_loaders(
    n_train_games=100, n_val_games=100, batch_size=32, seed=999
)

# Run inference and capture residuals
print("\n=== RESIDUAL STREAM ANALYSIS ===")
with torch.no_grad():
    batch = next(iter(val_loader))
    moves, legal_targets, next_move = batch
    
    logits = model(moves)
    
    # Get cached residuals
    resid_final = model.cache.get('resid_final')
    print(f"Residual shape: {resid_final.shape}")  # [batch, seq, d_model]
    
    # Analyze last position residual
    last_resid = resid_final[:, -1, :]  # [batch, d_model]
    print(f"Last position residual: mean={last_resid.mean():.4f}, std={last_resid.std():.4f}")
    print(f"Residual norm: {last_resid.norm(dim=-1).mean():.4f}")
    
    # Check predictions
    preds = logits[:, -1, :].argmax(dim=-1)
    is_legal = legal_targets[torch.arange(len(preds)), preds]
    acc = is_legal.float().mean()
    print(f"\nBatch accuracy: {acc*100:.2f}%")

print("\n=== MODEL READY FOR PROBE EXPERIMENTS ===")
