
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime

# Local imports
sys.path.append(os.getcwd())  # Ensure root is in path
from clean_audit.experiments.exp_a_foundation import SimpleTransformer, ModularArithmeticDataset, make_jamming_hook
from clean_audit.lib.metrics import AllostasisAudit

# Configuration
CONFIG = {
    'p': 113,           # Modular arithmetic prime
    'd_model': 32,      # REDUCED from 128 to force geometric crowding (n=113 > d=32)
    'vocab_size': 128,  # Match modulus roughly
    'n_epochs': 50,     # Enough to converge
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seeds': [42]       # Add more for statistical robustness
}

def train_model(model_name, noise_scale=0.0):
    """Train a model with specified noise injection."""
    print(f"\nTraining {model_name} (noise_scale={noise_scale})...")
    
    device = CONFIG['device']
    p = CONFIG['p']
    
    # Dataset
    train_dataset = ModularArithmeticDataset(p=p, seq_len=16, train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Model
    model = SimpleTransformer(
        vocab_size=CONFIG['vocab_size'],
        d_model=CONFIG['d_model'],
        n_heads=4,
        max_seq_len=16,
        disable_ffn=False
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    # Noise hook
    if noise_scale > 0:
        hook = make_jamming_hook(noise_scale)
        # We need to register this hook. SimpleTransformer doesn't have a hook point easily accessible
        # without using register_forward_hook on ln1.
        # Let's inspect exp_a_foundation.py to see how it applies hook.
        # It uses: model.ln1.register_forward_hook(hook)
        model.ln1.register_forward_hook(hook)
        print(f"  [Hook] Registered noise hook (scale={noise_scale}) at ln1")

    # Train loop
    model.train()
    for epoch in range(CONFIG['n_epochs']):
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            
            # Predict last token
            last_logits = logits[:, -1, :]
            targets = y[:, -1]
            
            loss = loss_fn(last_logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = last_logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        acc = correct / total
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{CONFIG['n_epochs']} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f}")
            
    # Save model
    save_path = f"{model_name}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Saved to {save_path}")
    return model

def evaluate_robustness(model, model_name):
    """Evaluate model against a sweep of noise scales."""
    print(f"\nEvaluating robustness for {model_name}...")
    device = CONFIG['device']
    p = CONFIG['p']
    
    val_dataset = ModularArithmeticDataset(p=p, seq_len=16, train=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    auditor = AllostasisAudit()
    
    # Noise sweep
    noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    results = []
    
    # Measure baseline A_activation (clean)
    model.eval()
    clean_amp = 0
    with torch.no_grad():
        # Clean pass (disable hooks if any)
        # We need a clean instance or remove hooks. 
        # Easier to reload weights into a fresh model to be sure.
        clean_model = SimpleTransformer(
            vocab_size=CONFIG['vocab_size'],
            d_model=CONFIG['d_model'],
            n_heads=4,
            max_seq_len=16,
            disable_ffn=False
        ).to(device)
        clean_model.load_state_dict(model.state_dict())
        clean_model.eval()
        
        # Helper to run validation
        def run_val(m):
            correct = 0
            total = 0
            amplitudes = []
            
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = m(x)
                last_logits = logits[:, -1, :]
                targets = y[:, -1]
                
                preds = last_logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
                if 'resid_post' in m.cache:
                    amp = auditor.compute_amplitude_activation(m.cache['resid_post'])
                    amplitudes.append(amp)
            
            return correct / total, sum(amplitudes)/len(amplitudes) if amplitudes else 0

        # Sweep
        for sigma in noise_levels:
            # Add noise hook
            # IMPORTANT: Reversible noise injection at test time
            # We inject at ln1 same as training
            sweep_model = SimpleTransformer(
                vocab_size=CONFIG['vocab_size'],
                d_model=CONFIG['d_model'],
                n_heads=4,
                max_seq_len=16,
                disable_ffn=False
            ).to(device)
            sweep_model.load_state_dict(model.state_dict())
            sweep_model.eval()
            
            if sigma > 0:
                hook = make_jamming_hook(sigma)
                sweep_model.ln1.register_forward_hook(hook)
            
            acc, amp = run_val(sweep_model)
            print(f"  σ={sigma:<3} | Acc: {acc:.4f} | A: {amp:.2f}")
            results.append({
                'model': model_name,
                'noise_sigma': sigma,
                'accuracy': acc,
                'A_activation': amp
            })
            
    return results

def compute_geometry_similarity(model_base, model_hard):
    """Compute cosine similarity of representations on clean data."""
    print("\nComputing geometry similarity (Q1 vs Q3)...")
    device = CONFIG['device']
    p = CONFIG['p']
    
    val_dataset = ModularArithmeticDataset(p=p, seq_len=16, train=False)
    # Just take one large batch
    loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    batch_x, _ = next(iter(loader))
    batch_x = batch_x.to(device)
    
    with torch.no_grad():
        _ = model_base(batch_x)
        rep_base = model_base.cache['resid_post'] # [B, T, D]
        
        _ = model_hard(batch_x)
        rep_hard = model_hard.cache['resid_post']
        
        # Flatten
        flat_base = rep_base.reshape(-1, CONFIG['d_model'])
        flat_hard = rep_hard.reshape(-1, CONFIG['d_model'])
        
        # Cosine similarity
        cos_sim = nn.functional.cosine_similarity(flat_base, flat_hard, dim=-1).mean().item()
        
    print(f"  Cosine Similarity (resid_post): {cos_sim:.4f}")
    return cos_sim

def main():
    # 1. Train Baseline (Clean)
    model_baseline = train_model("baseline_q1_q2", noise_scale=0.0)
    
    # 2. Train Hardened (Injured)
    model_hardened = train_model("hardened_q3_q4", noise_scale=2.0)
    
    # 3. Evaluate Protocols
    results = []
    results.extend(evaluate_robustness(model_baseline, "Baseline (Q1/Q2)"))
    results.extend(evaluate_robustness(model_hardened, "Hardened (Q3/Q4)"))
    
    # 4. Geometry Check
    sim = compute_geometry_similarity(model_baseline, model_hardened)
    
    # 5. Save Results
    df = pd.DataFrame(results)
    df.to_csv("injury_matrix_results.csv", index=False)
    print("\nResults saved to injury_matrix_results.csv")
    
    # Pivot for view
    pivot = df.pivot(index='noise_sigma', columns='model', values='accuracy')
    print("\nRobustness Matrix (Accuracy):")
    print(pivot)

if __name__ == "__main__":
    main()
