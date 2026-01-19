"""
Pre-generate Othello game caches for Phase 6 experiments.

Run this BEFORE training to avoid wasting GPU time on data generation.
This generates the precomputed .npz files with legal move vectors.

Usage:
    python scripts/pregenerate_othello_cache.py

    # Or for custom sizes:
    python scripts/pregenerate_othello_cache.py --n_train 100000 --n_val 10000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm
import argparse
import time

from lib.othello_dataset import OthelloGame, PASS_TOKEN, VOCAB_SIZE


def generate_games_with_precomputed_legal(
    n_games: int,
    seed: int = 42,
    max_moves: int = 60
):
    """
    Generate games with precomputed legal move vectors.

    Returns:
        moves_array: [n_games, max_moves] uint8
        lengths: [n_games] uint8
        legal_array: [total_samples, vocab_size] float32 (one per position)
    """
    np.random.seed(seed)

    moves_list = []
    lengths = []
    legal_list = []

    for i in tqdm(range(n_games), desc=f"Generating {n_games} games"):
        game = OthelloGame()
        game_moves = []
        game_legal = []

        while not game.game_over and len(game.move_history) < max_moves:
            # Store legal moves BEFORE making the move
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
        padded = game_moves + [PASS_TOKEN] * (max_moves - len(game_moves))
        moves_list.append(padded)
        lengths.append(len(game_moves))
        legal_list.extend(game_legal)

    moves_array = np.array(moves_list, dtype=np.uint8)
    lengths_array = np.array(lengths, dtype=np.uint8)
    legal_array = np.array(legal_list, dtype=np.float32)

    return moves_array, lengths_array, legal_array


def main():
    parser = argparse.ArgumentParser(description='Pre-generate Othello game caches')
    parser.add_argument('--n_train', type=int, default=100000, help='Number of training games')
    parser.add_argument('--n_val', type=int, default=10000, help='Number of validation games')
    parser.add_argument('--output_dir', type=str, default='clean_audit/data/othello_games',
                        help='Output directory for cache files')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("OTHELLO CACHE PRE-GENERATION")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Training games: {args.n_train:,}")
    print(f"Validation games: {args.n_val:,}")
    print()

    # Generate training cache
    train_path = output_dir / f"train_{args.n_train}_precomputed.npz"
    if train_path.exists():
        print(f"Training cache already exists: {train_path}")
        data = np.load(train_path)
        print(f"  Games: {len(data['lengths']):,}")
        print(f"  Legal vectors: {len(data['legal']):,}")
    else:
        print(f"Generating training cache...")
        start = time.time()
        moves, lengths, legal = generate_games_with_precomputed_legal(
            args.n_train, seed=args.seed
        )
        elapsed = time.time() - start

        np.savez_compressed(
            train_path,
            moves=moves,
            lengths=lengths,
            legal=legal
        )

        file_size = train_path.stat().st_size / 1024**2
        print(f"  Saved to: {train_path}")
        print(f"  File size: {file_size:.1f} MB")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
        print(f"  Games: {len(lengths):,}")
        print(f"  Legal vectors: {len(legal):,}")

    print()

    # Generate validation cache
    val_path = output_dir / f"val_{args.n_val}_precomputed.npz"
    if val_path.exists():
        print(f"Validation cache already exists: {val_path}")
        data = np.load(val_path)
        print(f"  Games: {len(data['lengths']):,}")
        print(f"  Legal vectors: {len(data['legal']):,}")
    else:
        print(f"Generating validation cache...")
        start = time.time()
        moves, lengths, legal = generate_games_with_precomputed_legal(
            args.n_val, seed=args.seed + 1000  # Different seed for val
        )
        elapsed = time.time() - start

        np.savez_compressed(
            val_path,
            moves=moves,
            lengths=lengths,
            legal=legal
        )

        file_size = val_path.stat().st_size / 1024**2
        print(f"  Saved to: {val_path}")
        print(f"  File size: {file_size:.1f} MB")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Games: {len(lengths):,}")
        print(f"  Legal vectors: {len(legal):,}")

    print()
    print("="*60)
    print("CACHE GENERATION COMPLETE")
    print("="*60)
    print()
    print("You can now run training with:")
    print(f"  python experiments/phase_6_othello/exp_o0_baseline_optimized.py")


if __name__ == '__main__':
    main()
