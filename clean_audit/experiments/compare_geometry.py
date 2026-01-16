
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports
sys.path.append(os.getcwd())  # Ensure root is in path
from clean_audit.experiments.exp_a_foundation import SimpleTransformer, ModularArithmeticDataset

def compare_geometry():
    # Config
    p = 113
    d_model = 32  # Crowded regime
    vocab_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load models
    def load_model(path):
        model = SimpleTransformer(
            vocab_size=vocab_size, d_model=d_model, n_heads=4, max_seq_len=16, disable_ffn=False
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model

    print("Loading models...")
    baseline = load_model('baseline_q1_q2.pt')
    hardened = load_model('hardened_q3_q4.pt')
    
    # Get data
    dataset = ModularArithmeticDataset(p=p, seq_len=16, train=False)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    # Compare representations
    print("Computing Cosine Similarity of resid_post...")
    sims = []
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            
            _ = baseline(x)
            rep_base = baseline.cache['resid_post'] # [B, T, D]
            
            _ = hardened(x)
            rep_hard = hardened.cache['resid_post']
            
            # Flatten to [B*T, D]
            flat_base = rep_base.reshape(-1, d_model)
            flat_hard = rep_hard.reshape(-1, d_model)
            
            # Cosine sim per vector
            sim = nn.functional.cosine_similarity(flat_base, flat_hard, dim=-1)
            sims.append(sim.mean().item())
            
    mean_sim = sum(sims) / len(sims)
    print(f"\nMean Cosine Similarity (Q1 vs Q3): {mean_sim:.4f}")
    
    # Also check if representations are just scaled versions
    # Compare normalized vectors
    print("Checking alignment of normalized vectors...")
    
    return mean_sim

if __name__ == "__main__":
    compare_geometry()
