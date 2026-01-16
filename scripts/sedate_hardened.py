
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from clean_audit.experiments.exp_a_foundation import SimpleTransformer, ModularArithmeticDataset, make_naive_clamp_hook

# Local imports
sys.path.append(os.getcwd())

def sedate_hardened():
    print("THE SMOKING GUN TEST: Sedating the Hardened Model")
    
    # Config
    p = 113
    d_model = 32
    max_seq_len = 16
    vocab_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Hardened Model
    model = SimpleTransformer(vocab_size=128, d_model=32, n_heads=4, max_seq_len=16, disable_ffn=False).to(device)
    model.load_state_dict(torch.load('hardened_q3_q4.pt', map_location=device))
    model.eval()
    
    # 2. Setup Sedation (Clamp + Noise)
    # Target Norm: 3.0 (Aggressive sedation, ~70% of natural 4.33)
    target_norm = 3.0
    noise_sigma = 5.0 # Godzilla level
    
    print(f"Condition: | Hardened Model | Clamp A -> {target_norm} | Noise sigma={noise_sigma} |")
    
    # Register Clamp Hook (Naive) - Before LN_final? 
    # Actually exp_a_foundation applies clamp_fn if set.
    # But we want to modify the loaded model.
    # SimpleTransformer has 'clamp_fn'.
    from clean_audit.lib.clamps import NaiveClamp
    model.clamp_fn = NaiveClamp(target_norm, layer_idx=-1)
    
    # Register Noise Hook (Post Clamp)
    # SimpleTransformer applies post_clamp_noise_scale if set.
    model.post_clamp_noise_scale = noise_sigma
    
    # 3. Evaluate
    dataset = ModularArithmeticDataset(p=p, seq_len=16, train=False)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            last_logits = logits[:, -1, :]
            targets = y[:, -1]
            
            preds = last_logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    acc = correct / total
    print(f"\nFinal Accuracy: {acc*100:.2f}%")
    
    if acc > 0.90:
        print(">> RESULT: SURVIVAL. Geometry replaced Amplitude completely.")
    else:
        print(">> RESULT: DEATH. Amplitude margin was still needed.")

if __name__ == "__main__":
    sedate_hardened()
