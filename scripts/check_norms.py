
import torch
import os
import sys
from clean_audit.experiments.exp_a_foundation import SimpleTransformer, ModularArithmeticDataset

def check():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p = 113
    d = 32
    
    def get_norm(path):
        model = SimpleTransformer(vocab_size=128, d_model=32, n_heads=4, max_seq_len=16).to(device)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        
        dataset = ModularArithmeticDataset(p=p, seq_len=16, train=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024)
        
        norms = []
        with torch.no_grad():
            for x,y in loader:
                x = x.to(device)
                _ = model(x)
                # resid_post is Pre-LN in SimpleTransformer
                resid_pre_ln = model.cache['resid_post']
                
                # Apply LN manual
                resid_post_ln = model.ln_final(resid_pre_ln)
                
                norm = resid_post_ln.norm(dim=-1).mean().item()
                norms.append(norm)
                if len(norms) > 5: break
        return sum(norms)/len(norms)

    print(f"Baseline A: {get_norm('baseline_q1_q2.pt'):.4f}")
    print(f"Hardened A: {get_norm('hardened_q3_q4.pt'):.4f}")

if __name__ == '__main__':
    check()
