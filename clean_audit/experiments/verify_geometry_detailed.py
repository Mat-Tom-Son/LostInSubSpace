
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Local imports
sys.path.append(os.getcwd())
from clean_audit.experiments.exp_a_foundation import SimpleTransformer, ModularArithmeticDataset

def verify_geometry():
    # Config (Crowded Regime)
    p = 113
    d_model = 32
    max_seq_len = 16
    vocab_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Verifying Geometry (p={p}, d={d_model})...")
    
    # Load Models
    def load(path):
        m = SimpleTransformer(vocab_size=128, d_model=32, n_heads=4, max_seq_len=16, disable_ffn=False).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        return m

    baseline = load('baseline_q1_q2.pt')
    hardened = load('hardened_q3_q4.pt')
    
    # Dataset
    data = ModularArithmeticDataset(p=p, seq_len=16, train=False)
    loader = DataLoader(data, batch_size=512, shuffle=False)
    
    # Accumulators
    cos_sims_resid = []
    attn_diffs = []
    
    # Iterate
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            
            # Forward Baseline
            _ = baseline(x)
            res_b = baseline.cache['resid_post']       # [B, T, D]
            attn_b = baseline.cache['attn_weights']    # [B, H, T, T]
            
            # Forward Hardened
            _ = hardened(x)
            res_h = hardened.cache['resid_post']
            attn_h = hardened.cache['attn_weights']
            
            # 1. Residual Cosine Similarity (Per token)
            flat_b = res_b.reshape(-1, d_model)
            flat_h = res_h.reshape(-1, d_model)
            sim = nn.functional.cosine_similarity(flat_b, flat_h, dim=-1)
            cos_sims_resid.append(sim.mean().item())
            
            # 2. Attention Pattern Correlation (Per head)
            # Flatten to [B, H, T*T]
            at_b = attn_b.reshape(attn_b.shape[0], attn_b.shape[1], -1)
            at_h = attn_h.reshape(attn_h.shape[0], attn_h.shape[1], -1)
            
            # Cosine sim per head
            # [B, H, T*T] -> dot over T*T
            head_sims = nn.functional.cosine_similarity(at_b, at_h, dim=-1) # [B, H]
            attn_diffs.append(head_sims.mean(dim=0).cpu().numpy()) # Avg over batch -> [H]

    print("-" * 40)
    print("RESULTS")
    print("-" * 40)
    
    # Report Residual Sim
    final_res_sim = np.mean(cos_sims_resid)
    print(f"Residual Stream Cos Similarity: {final_res_sim:.4f}")
    if abs(final_res_sim) < 0.1:
        print(">> ORTHOGONAL GEOMETRY CONFIRMED")
    
    # Report Attention Sim
    mean_head_sims = np.mean(attn_diffs, axis=0)
    print("\nAttention Pattern Similarity (Per Head):")
    for h, s in enumerate(mean_head_sims):
        print(f"  Head {h}: {s:.4f}")
    
    avg_attn_sim = np.mean(mean_head_sims)
    print(f"\nMean Attn Sim: {avg_attn_sim:.4f}")

    if avg_attn_sim < 0.5:
         print(">> ATTENTION MECHANISM REWIRED")

if __name__ == "__main__":
    verify_geometry()
