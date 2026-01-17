
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import sys
import os

# Local imports (running from clean_audit/)

from experiments.exp_a_foundation import SimpleTransformer, ModularArithmeticDataset
from lib.metrics import AllostasisAudit

def measure_amplitude(modulus, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Measuring p={modulus} from {model_path}...")
    
    # Config (must match training)
    d_model = 128
    # Vocab size varies by modulus if we followed the sweep args.
    # p=7 -> vocab=16, p=13 -> vocab=32, p=41 -> vocab=64, p=113 -> 128, p=227 -> 256
    # Let's infer or try to load.
    
    vocab_map = {
        7: 16,
        13: 32,
        41: 64,
        113: 128,
        227: 256
    }
    vocab_size = vocab_map.get(modulus, 128)
    
    # Create dataset
    val_dataset = ModularArithmeticDataset(p=modulus, seq_len=16, train=False)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # Create model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        max_seq_len=128,
        disable_ffn=False
    ).to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return None

    model.eval()
    auditor = AllostasisAudit()
    
    amplitudes = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            
            # Get residual stream
            if 'resid_post' in model.cache:
                resid = model.cache['resid_post']
                A_act = auditor.compute_amplitude_activation(resid)
                amplitudes.append(A_act)
    
    avg_amp = sum(amplitudes) / len(amplitudes)
    return avg_amp

def main():
    moduli = [7, 13, 41, 113, 227]
    results = {}
    
    print("-" * 40)
    print(f"{'Modulus (p)':<12} | {'A_activation':<12}")
    print("-" * 40)
    
    for p in moduli:
        path = f"../modular_p{p}.pt"  # Look in parent dir
        if not os.path.exists(path):
            print(f"Skipping p={p}: {path} not found")
            continue
            
        amp = measure_amplitude(p, path)
        if amp is not None:
            results[p] = amp
            print(f"{p:<12} | {amp:.4f}")
            
    print("-" * 40)

if __name__ == "__main__":
    main()
