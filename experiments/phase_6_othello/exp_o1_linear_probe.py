"""
Phase 6 Experiment O.1: Linear Probe for Board State

Train a linear probe on residual stream to predict board state (Mine/Theirs/Empty).
This characterizes the "S" vector - the world model representation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, 'clean_audit')
from lib.deep_transformer import DeepTransformer
from lib.othello_dataset import OthelloDataset, OthelloGame, VOCAB_SIZE, PASS_TOKEN
from tqdm import tqdm


def get_board_state_from_moves(moves):
    """Replay moves and return board state (64,) with 0=empty, 1=mine, 2=theirs."""
    game = OthelloGame()
    for m in moves:
        if m != PASS_TOKEN:
            game.make_move(int(m))
    return game.get_board_state()  # (8, 8, 3) one-hot


def collect_probe_data(model, dataset, n_samples=5000, layer_idx=3, device='cpu'):
    """
    Collect (residual, board_state) pairs for probe training.
    
    Args:
        model: Trained Othello model
        dataset: OthelloDataset
        n_samples: Number of samples to collect
        layer_idx: Which layer's residual to probe (0-3 for 4L model)
    
    Returns:
        residuals: [n_samples, d_model]
        board_states: [n_samples, 64] with values 0/1/2
    """
    model.eval()
    model.to(device)
    
    residuals = []
    board_states = []
    
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    for idx in tqdm(indices, desc="Collecting probe data"):
        moves, legal, next_move = dataset[idx]
        
        # Get residual from model
        with torch.no_grad():
            logits = model(moves.unsqueeze(0).to(device))
            
            # Get residual from specified layer (post-block)
            resid_key = f'blocks.{layer_idx}.resid_post'
            resid = model.cache.get(resid_key)
            if resid is None:
                resid = model.cache.get('resid_final')
            
            # Take last position
            last_resid = resid[0, -1, :].cpu()  # [d_model]
        
        # Get board state by replaying
        move_list = moves.tolist()
        # Remove padding
        move_list = [m for m in move_list if m != PASS_TOKEN or move_list.index(m) == len(move_list) - 1]
        
        game = OthelloGame()
        for m in move_list:
            if m != PASS_TOKEN:
                game.make_move(int(m))
        
        # Get board state: 0=empty, 1=current player, 2=opponent
        board = game.board.copy()  # (8, 8)
        current = game.current_player
        opponent = 3 - current  # 1->2, 2->1
        
        # Remap to relative: 0=empty, 1=mine, 2=theirs
        state = np.zeros(64, dtype=np.int64)
        for i in range(64):
            r, c = i // 8, i % 8
            if board[r, c] == current:
                state[i] = 1  # Mine
            elif board[r, c] == opponent:
                state[i] = 2  # Theirs
            # else 0 = empty
        
        residuals.append(last_resid)
        board_states.append(state)
    
    return torch.stack(residuals), torch.tensor(np.array(board_states))


class BoardProbe(nn.Module):
    """Linear probe: residual -> board state (64 squares × 3 classes)."""
    
    def __init__(self, d_model=512, n_squares=64, n_classes=3):
        super().__init__()
        self.probe = nn.Linear(d_model, n_squares * n_classes)
        self.n_squares = n_squares
        self.n_classes = n_classes
    
    def forward(self, x):
        # x: [batch, d_model]
        out = self.probe(x)  # [batch, 64*3]
        return out.view(-1, self.n_squares, self.n_classes)  # [batch, 64, 3]


def train_probe(residuals, board_states, d_model=512, epochs=50, lr=1e-3):
    """Train linear probe on collected data."""
    probe = BoardProbe(d_model=d_model)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    
    # Split train/val
    n = len(residuals)
    split = int(0.8 * n)
    train_r, val_r = residuals[:split], residuals[split:]
    train_b, val_b = board_states[:split], board_states[split:]
    
    best_acc = 0
    
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        
        logits = probe(train_r)  # [n, 64, 3]
        loss = F.cross_entropy(logits.view(-1, 3), train_b.view(-1))
        loss.backward()
        optimizer.step()
        
        # Validate
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_r)
            val_preds = val_logits.argmax(dim=-1)  # [n, 64]
            acc = (val_preds == val_b).float().mean()
            
            # Per-class accuracy
            empty_mask = val_b == 0
            mine_mask = val_b == 1
            theirs_mask = val_b == 2
            
            empty_acc = (val_preds[empty_mask] == 0).float().mean() if empty_mask.sum() > 0 else 0
            mine_acc = (val_preds[mine_mask] == 1).float().mean() if mine_mask.sum() > 0 else 0
            theirs_acc = (val_preds[theirs_mask] == 2).float().mean() if theirs_mask.sum() > 0 else 0
        
        if acc > best_acc:
            best_acc = acc
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc*100:.2f}% (E:{empty_acc*100:.1f}% M:{mine_acc*100:.1f}% T:{theirs_acc*100:.1f}%)")
    
    return probe, best_acc


def main():
    print("=" * 60)
    print("PHASE 6 O.1: LINEAR PROBE FOR BOARD STATE")
    print("=" * 60)
    
    # Load model
    ckpt_path = 'data/othello_games/LostInSubSpace_clean_audit_clean_audit_data_othello_baseline_20260116_195406_final.pt'
    d = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    arch = d.get('architecture', {})
    
    model = DeepTransformer(
        vocab_size=arch.get('vocab_size', 65),
        d_model=arch.get('d_model', 512),
        n_heads=arch.get('n_heads', 8),
        n_layers=arch.get('n_layers', 4),
        max_seq_len=arch.get('ctx_len', 60)
    )
    model.load_state_dict(d['model_state_dict'])
    print(f"Loaded model: {d['final_metrics']['legal_acc']*100:.2f}% accuracy")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = OthelloDataset(n_games=2000, seed=123)
    
    # Collect probe data
    print("\nCollecting probe training data...")
    residuals, board_states = collect_probe_data(model, dataset, n_samples=5000, layer_idx=3)
    print(f"Collected {len(residuals)} samples")
    print(f"Residual shape: {residuals.shape}")
    print(f"Board state distribution: Empty={torch.sum(board_states==0)}, Mine={torch.sum(board_states==1)}, Theirs={torch.sum(board_states==2)}")
    
    # Train probe
    print("\nTraining linear probe...")
    probe, best_acc = train_probe(residuals, board_states, d_model=arch.get('d_model', 512))
    
    print("\n" + "=" * 60)
    print(f"PROBE ACCURACY: {best_acc*100:.2f}%")
    print("=" * 60)
    
    if best_acc > 0.8:
        print("✅ World model representation found in residual stream!")
    else:
        print("⚠️ Probe accuracy lower than expected")
    
    # Save probe
    torch.save(probe.state_dict(), 'data/board_probe.pt')
    print("\nProbe saved to data/board_probe.pt")


if __name__ == '__main__':
    main()
