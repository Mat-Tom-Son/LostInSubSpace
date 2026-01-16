"""
Othello Dataset for Phase 6 Experiments

Provides:
1. Othello game simulator (legal move generation)
2. Dataset of synthetic games (random legal moves)
3. PyTorch DataLoader compatible with G×S experiments

Based on the format used by Neel Nanda's Othello-GPT work.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Set
from tqdm import tqdm
import json
from pathlib import Path


# Board constants
BOARD_SIZE = 8
EMPTY = 0
BLACK = 1
WHITE = 2

# Move encoding: 0-63 = squares (row-major), 64 = pass
PASS_TOKEN = 64
VOCAB_SIZE = 65  # 64 squares + pass


def pos_to_idx(row: int, col: int) -> int:
    """Convert (row, col) to move index 0-63."""
    return row * 8 + col


def idx_to_pos(idx: int) -> Tuple[int, int]:
    """Convert move index to (row, col)."""
    return idx // 8, idx % 8


class OthelloGame:
    """
    Othello game simulator with legal move computation.
    
    Board state:
    - 0 = empty
    - 1 = black
    - 2 = white
    
    Black moves first.
    """
    
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        # Starting position
        self.board[3, 3] = WHITE
        self.board[3, 4] = BLACK
        self.board[4, 3] = BLACK
        self.board[4, 4] = WHITE
        self.current_player = BLACK
        self.move_history: List[int] = []
        self.game_over = False
    
    def copy(self) -> 'OthelloGame':
        """Create a deep copy of the game state."""
        new_game = OthelloGame.__new__(OthelloGame)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        new_game.game_over = self.game_over
        return new_game
    
    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8
    
    def _get_flips(self, row: int, col: int, player: int) -> List[Tuple[int, int]]:
        """Get all pieces that would be flipped by playing at (row, col)."""
        if self.board[row, col] != EMPTY:
            return []
        
        opponent = WHITE if player == BLACK else BLACK
        all_flips = []
        
        for dr, dc in self.DIRECTIONS:
            flips = []
            r, c = row + dr, col + dc
            
            # Move in direction while finding opponent pieces
            while self._in_bounds(r, c) and self.board[r, c] == opponent:
                flips.append((r, c))
                r, c = r + dr, c + dc
            
            # Valid flip if we end on our own piece
            if flips and self._in_bounds(r, c) and self.board[r, c] == player:
                all_flips.extend(flips)
        
        return all_flips
    
    def get_legal_moves(self) -> Set[int]:
        """Get set of legal move indices for current player."""
        legal = set()
        for row in range(8):
            for col in range(8):
                if self._get_flips(row, col, self.current_player):
                    legal.add(pos_to_idx(row, col))
        return legal
    
    def is_legal(self, move_idx: int) -> bool:
        """Check if a move is legal."""
        if move_idx == PASS_TOKEN:
            return len(self.get_legal_moves()) == 0
        row, col = idx_to_pos(move_idx)
        return len(self._get_flips(row, col, self.current_player)) > 0
    
    def make_move(self, move_idx: int) -> bool:
        """
        Make a move and update the board.
        
        Returns True if successful, False if illegal.
        """
        if self.game_over:
            return False
        
        if move_idx == PASS_TOKEN:
            # Pass is only legal if no other moves
            if self.get_legal_moves():
                return False
            self.move_history.append(PASS_TOKEN)
            self.current_player = WHITE if self.current_player == BLACK else BLACK
            # Check for game over (both players pass)
            if len(self.move_history) >= 2 and self.move_history[-2] == PASS_TOKEN:
                self.game_over = True
            return True
        
        row, col = idx_to_pos(move_idx)
        flips = self._get_flips(row, col, self.current_player)
        
        if not flips:
            return False
        
        # Place piece and flip
        self.board[row, col] = self.current_player
        for r, c in flips:
            self.board[r, c] = self.current_player
        
        self.move_history.append(move_idx)
        self.current_player = WHITE if self.current_player == BLACK else BLACK
        
        # Check for game over
        if self.board.sum() == 0 or len(self.move_history) >= 60:
            self.game_over = True
        elif not self.get_legal_moves():
            # Current player must pass, check if opponent can move
            self.current_player = WHITE if self.current_player == BLACK else BLACK
            if not self.get_legal_moves():
                self.game_over = True
            self.current_player = WHITE if self.current_player == BLACK else BLACK
        
        return True
    
    def get_board_state(self) -> np.ndarray:
        """
        Get board state relative to current player.
        
        Returns (8, 8, 3) one-hot: [empty, mine, theirs]
        """
        mine = self.current_player
        theirs = WHITE if mine == BLACK else BLACK
        
        state = np.zeros((8, 8, 3), dtype=np.float32)
        state[:, :, 0] = (self.board == EMPTY).astype(np.float32)
        state[:, :, 1] = (self.board == mine).astype(np.float32)
        state[:, :, 2] = (self.board == theirs).astype(np.float32)
        
        return state
    
    def get_legal_moves_vector(self) -> np.ndarray:
        """Get binary vector of legal moves (length 61)."""
        legal = np.zeros(VOCAB_SIZE, dtype=np.float32)
        moves = self.get_legal_moves()
        if moves:
            for m in moves:
                legal[m] = 1.0
        else:
            legal[PASS_TOKEN] = 1.0
        return legal


def generate_random_game_with_legal(max_moves: int = 60) -> Tuple[List[int], List[List[float]]]:
    """
    Generate a single random game by playing random legal moves.
    Also stores legal moves at each position for O(1) dataset access.
    
    Returns:
        (move_history, legal_moves_at_each_step) - all Python native types for JSON
    """
    game = OthelloGame()
    legal_at_step = []
    
    while not game.game_over and len(game.move_history) < max_moves:
        # Store legal moves BEFORE making the move
        legal_at_step.append(game.get_legal_moves_vector().tolist())
        
        legal = list(game.get_legal_moves())
        if not legal:
            game.make_move(PASS_TOKEN)
        else:
            move = int(np.random.choice(legal))  # Convert to Python int for JSON
            game.make_move(move)
    
    # Convert move history to Python ints
    moves = [int(m) for m in game.move_history]
    return moves, legal_at_step


def generate_games_with_legal(n_games: int, seed: int = 42, show_progress: bool = True) -> List[dict]:
    """
    Generate n random games with pre-computed legal moves.
    
    Returns list of {moves: [...], legal: [[...], ...]}
    """
    np.random.seed(seed)
    games = []
    
    iterator = range(n_games)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating games")
    
    for _ in iterator:
        moves, legal = generate_random_game_with_legal()
        games.append({'moves': moves, 'legal': legal})
    
    return games


class OthelloDatasetEfficient(Dataset):
    """
    Memory-efficient Othello dataset using numpy binary storage.
    
    Instead of loading all legal move vectors into RAM (which can be 10GB+),
    we store only moves and compute legal moves on-the-fly via replay.
    This trades CPU for RAM.
    
    For faster training, we also support pre-computed mode with numpy mmap.
    """
    
    def __init__(
        self,
        n_games: int = 100000,
        ctx_len: int = 60,
        min_moves: int = 4,
        seed: int = 42,
        cache_path: Optional[str] = None,
        precompute_legal: bool = False
    ):
        """
        Args:
            n_games: Number of games to generate
            ctx_len: Maximum context length
            min_moves: Minimum moves before sampling positions
            seed: Random seed
            cache_path: Path to cache games (optional, .npz format)
            precompute_legal: If True, compute legal moves during generation (slower but faster training)
        """
        self.ctx_len = ctx_len
        self.min_moves = min_moves
        self.precompute_legal = precompute_legal
        
        # Games stored as [n_games, max_moves] uint8 array
        # Legal moves stored as [n_samples, vocab_size] if precomputed
        
        npz_path = cache_path.replace('.json', '.npz') if cache_path else None
        
        if npz_path and Path(npz_path).exists():
            print(f"Loading games from cache: {npz_path}")
            data = np.load(npz_path, allow_pickle=True)
            self.moves_array = data['moves']  # [n_games, max_len]
            self.lengths = data['lengths']     # [n_games]
            self.legal_array = data.get('legal', None)  # Optional [n_samples, vocab]
            print(f"  Loaded {len(self.lengths)} games ({self.moves_array.nbytes / 1024 / 1024:.1f} MB)")
        else:
            print(f"Generating {n_games} Othello games...")
            self._generate_and_cache(n_games, seed, npz_path, precompute_legal)
        
        # Build sample index
        self._build_index()
        print(f"  Total samples: {len(self.samples)}")
    
    def _generate_and_cache(self, n_games: int, seed: int, npz_path: Optional[str], precompute: bool):
        """Generate games and store efficiently."""
        np.random.seed(seed)
        
        max_len = 60
        moves_list = []
        lengths = []
        legal_list = [] if precompute else None
        
        for i in tqdm(range(n_games), desc="Generating games"):
            game = OthelloGame()
            game_moves = []
            game_legal = [] if precompute else None
            
            while not game.game_over and len(game.move_history) < max_len:
                if precompute:
                    game_legal.append(game.get_legal_moves_vector())
                
                legal = list(game.get_legal_moves())
                if not legal:
                    game.make_move(PASS_TOKEN)
                    game_moves.append(PASS_TOKEN)
                else:
                    move = int(np.random.choice(legal))
                    game.make_move(move)
                    game_moves.append(move)
            
            # Pad to fixed length
            padded = game_moves + [PASS_TOKEN] * (max_len - len(game_moves))
            moves_list.append(padded)
            lengths.append(len(game_moves))
            
            if precompute and game_legal:
                legal_list.extend(game_legal)
        
        # Convert to numpy arrays
        self.moves_array = np.array(moves_list, dtype=np.uint8)
        self.lengths = np.array(lengths, dtype=np.uint8)
        
        if precompute:
            self.legal_array = np.array(legal_list, dtype=np.float32)
        else:
            self.legal_array = None
        
        # Save to cache
        if npz_path:
            Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
            save_dict = {'moves': self.moves_array, 'lengths': self.lengths}
            if self.legal_array is not None:
                save_dict['legal'] = self.legal_array
            np.savez_compressed(npz_path, **save_dict)
            file_size = Path(npz_path).stat().st_size / 1024 / 1024
            print(f"  Cached to {npz_path} ({file_size:.1f} MB)")
    
    def _build_index(self):
        """Build sample index mapping idx -> (game_idx, position)."""
        self.samples = []
        self.legal_offsets = []  # For precomputed legal arrays
        offset = 0
        
        for game_idx in range(len(self.lengths)):
            game_len = int(self.lengths[game_idx])
            for pos in range(self.min_moves, game_len):
                self.samples.append((game_idx, pos))
                if self.legal_array is not None:
                    self.legal_offsets.append(offset + pos)
            offset += game_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        game_idx, pos = self.samples[idx]
        
        # Get moves up to position
        moves = self.moves_array[game_idx, :pos].tolist()
        
        # Pad to ctx_len
        if len(moves) < self.ctx_len:
            padded = [PASS_TOKEN] * (self.ctx_len - len(moves)) + moves
        else:
            padded = moves[-self.ctx_len:]
        
        # Get legal moves (precomputed or replay)
        if self.legal_array is not None and idx < len(self.legal_offsets):
            legal_vector = self.legal_array[self.legal_offsets[idx]]
        else:
            # Replay game to compute legal moves (slower but memory efficient)
            game = OthelloGame()
            for m in moves:
                game.make_move(int(m))
            legal_vector = game.get_legal_moves_vector()
        
        # Next move target
        next_move = int(self.moves_array[game_idx, pos])
        
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(legal_vector, dtype=torch.float32),
            torch.tensor(next_move, dtype=torch.long)
        )


# Keep old class name for compatibility but use efficient version
OthelloDataset = OthelloDatasetEfficient


def get_othello_loaders(
    n_train_games: int = 100000,
    n_val_games: int = 10000,
    batch_size: int = 128,
    ctx_len: int = 60,
    seed: int = 42,
    cache_dir: Optional[str] = "clean_audit/data/othello_games",
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders.
    
    Returns:
        (train_loader, val_loader)
    """
    cache_train = f"{cache_dir}/train_{n_train_games}.json" if cache_dir else None
    cache_val = f"{cache_dir}/val_{n_val_games}.json" if cache_dir else None
    
    train_dataset = OthelloDataset(
        n_games=n_train_games,
        ctx_len=ctx_len,
        seed=seed,
        cache_path=cache_train
    )
    
    val_dataset = OthelloDataset(
        n_games=n_val_games,
        ctx_len=ctx_len,
        seed=seed + 1000,  # Different seed for validation
        cache_path=cache_val
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Quick test
    print("Testing OthelloGame...")
    game = OthelloGame()
    print(f"Initial legal moves: {game.get_legal_moves()}")
    
    # Play a few random moves
    for _ in range(5):
        legal = list(game.get_legal_moves())
        if legal:
            move = np.random.choice(legal)
            game.make_move(move)
            print(f"Made move {move}, new legal: {game.get_legal_moves()}")
    
    print(f"\nBoard state shape: {game.get_board_state().shape}")
    print(f"Legal moves vector shape: {game.get_legal_moves_vector().shape}")
    
    print("\nTesting OthelloDataset...")
    dataset = OthelloDataset(n_games=100)
    print(f"Dataset size: {len(dataset)}")
    
    moves, legal, next_move = dataset[0]
    print(f"Sample shapes: moves={moves.shape}, legal={legal.shape}, next={next_move.shape}")
