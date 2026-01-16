"""
TinyStories Dataset for Phase 5 Experiments

Provides a PyTorch Dataset for TinyStories language modeling,
compatible with the existing G×S experiment infrastructure.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
from typing import Optional, List
import numpy as np


class TinyStoriesDataset(Dataset):
    """
    TinyStories dataset for next-token prediction.
    
    Returns (input_tokens, target_tokens) pairs where:
    - input_tokens: tokens[:-1]
    - target_tokens: tokens[1:]
    
    This matches the format expected by the DeepTransformer for LM training.
    """
    
    def __init__(
        self, 
        split: str = 'train',
        max_length: int = 128,
        max_samples: Optional[int] = 100000,
        cache_tokenized: bool = True,
        seed: int = 42
    ):
        """
        Args:
            split: 'train' or 'validation'
            max_length: Maximum sequence length (including padding)
            max_samples: Limit dataset size for faster iteration (None for full)
            cache_tokenized: Pre-tokenize all data (faster training, more RAM)
            seed: Random seed for reproducible shuffling
        """
        self.max_length = max_length
        self.split = split
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.vocab_size = self.tokenizer.vocab_size
        
        # Load dataset
        print(f"Loading TinyStories ({split})...")
        dataset = load_dataset('roneneldan/TinyStories', split=split)
        
        # Shuffle and limit
        if max_samples and len(dataset) > max_samples:
            np.random.seed(seed)
            indices = np.random.permutation(len(dataset))[:max_samples]
            dataset = dataset.select(indices)
            print(f"  Using {max_samples} samples (shuffled with seed {seed})")
        
        # Pre-tokenize if caching
        if cache_tokenized:
            print(f"  Tokenizing {len(dataset)} stories...")
            self.data = []
            skipped = 0
            for item in tqdm(dataset, desc='Tokenizing'):
                # Skip empty stories
                text = item.get('text', '')
                if not text or len(text.strip()) < 10:
                    skipped += 1
                    continue
                    
                tokens = self.tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )['input_ids'].squeeze(0)
                
                # Verify shape is correct
                if tokens.shape[0] == max_length:
                    self.data.append(tokens)
                else:
                    skipped += 1
                    
            if skipped > 0:
                print(f"  Skipped {skipped} invalid stories")
            self.raw_dataset = None
        else:
            self.data = None
            self.raw_dataset = dataset
    
    def __len__(self):
        if self.data is not None:
            return len(self.data)
        return len(self.raw_dataset)
    
    def __getitem__(self, idx):
        if self.data is not None:
            tokens = self.data[idx]
        else:
            tokens = self.tokenizer(
                self.raw_dataset[idx]['text'],
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )['input_ids'].squeeze(0)
        
        # For language modeling: input is tokens[:-1], target is tokens[1:]
        return tokens[:-1], tokens[1:]


class TinyStoriesLMDataset(Dataset):
    """
    Alternative: Returns full sequences for more flexible loss computation.
    
    Returns (tokens, attention_mask) where loss is computed on tokens[1:]
    given tokens[:-1] as input.
    """
    
    def __init__(
        self,
        split: str = 'train',
        max_length: int = 128,
        max_samples: Optional[int] = 100000,
        seed: int = 42
    ):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self.max_length = max_length
        
        print(f"Loading TinyStories ({split})...")
        dataset = load_dataset('roneneldan/TinyStories', split=split)
        
        if max_samples and len(dataset) > max_samples:
            np.random.seed(seed)
            indices = np.random.permutation(len(dataset))[:max_samples]
            dataset = dataset.select(indices)
        
        print(f"  Tokenizing {len(dataset)} stories...")
        self.data = []
        for item in tqdm(dataset, desc='Tokenizing'):
            enc = self.tokenizer(
                item['text'],
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            self.data.append({
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['input_ids'], self.data[idx]['attention_mask']


def get_tinystories_loaders(
    batch_size: int = 32,
    max_length: int = 128,
    train_samples: int = 100000,
    val_samples: int = 10000,
    seed: int = 42,
    num_workers: int = 0
):
    """
    Convenience function to get train and validation dataloaders.
    
    Returns:
        (train_loader, val_loader, vocab_size, pad_token_id)
    """
    from torch.utils.data import DataLoader
    
    train_dataset = TinyStoriesDataset(
        split='train',
        max_length=max_length,
        max_samples=train_samples,
        seed=seed
    )
    
    val_dataset = TinyStoriesDataset(
        split='validation',
        max_length=max_length,
        max_samples=val_samples,
        seed=seed
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
    
    return (
        train_loader, 
        val_loader, 
        train_dataset.vocab_size,
        train_dataset.pad_token_id
    )


if __name__ == '__main__':
    # Quick test
    print("Testing TinyStoriesDataset...")
    dataset = TinyStoriesDataset(split='train', max_samples=100)
    print(f"Dataset size: {len(dataset)}")
    
    x, y = dataset[0]
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Vocab size: {dataset.vocab_size}")
    
    # Decode first example
    tokenizer = dataset.tokenizer
    print(f"\nFirst example (input tokens):")
    print(tokenizer.decode(x[:50]))
