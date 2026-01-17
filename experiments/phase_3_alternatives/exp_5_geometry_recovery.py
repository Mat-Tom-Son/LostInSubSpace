"""
EXPERIMENT 5: Forced Geometry Recovery (Modular Arithmetic)
===========================================================

Critique: "Geometry causality is shown via swap (crash), but not via recovery."
Goal: Prove (or disprove) that transplanting "Hardened" geometry onto a vulnerable baseline
      confers robustness (sufficiency & portability).

Refinement: Uses Modular Arithmetic (p=113) to ensure valid generalization (proven in Exp 2.2),
            avoiding the memorization pitfalls of the Repeat/Interleaved tasks.

Protocol:
1. Train BASELINE model (Standard CE) -> 100% Acc on (a+b)%p
2. Train HARDENED model (Noise Injection) -> 100% Acc + Robustness
3. Transplant Hardened QK -> Baseline (keeping Baseline S)
4. Evaluate Hybrid robustness.

Hypothesis:
    If G is fungible/universal: Hybrid recovers robustness (Recovery > 0.8).
    If G is coupled to S: Hybrid crashes (Recovery ~ 0.0).

Usage:
    python exp_5_geometry_recovery.py --n_seeds 3 --device cuda
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Tuple, List, Optional
import argparse
import json
import copy

from lib.deep_transformer import DeepModularTransformer
from lib.part_b_utils import swap_qk_parameters
from lib.logging_utils import setup_reproducibility

def layer_cosine_similarity(cache1: Dict[str, torch.Tensor], 
                            cache2: Dict[str, torch.Tensor], 
                            key: str) -> float:
    """Compute cosine similarity between cached tensors."""
    t1 = cache1[key].flatten()
    t2 = cache2[key].flatten()
    return F.cosine_similarity(t1, t2, dim=0).item()

# ==============================================================================
# DATASET
# ==============================================================================

class ModularAdditionDataset(Dataset):
    """Modular addition: (a + b) mod p"""
    
    def __init__(self, p: int = 113, split: str = 'train', train_frac: float = 0.5, seed: int = 42):
        self.p = p
        
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        # Deterministic shuffle based on seed
        rng = np.random.RandomState(seed)
        rng.shuffle(all_pairs)
        
        n_train = int(len(all_pairs) * train_frac)
        
        if split == 'train':
            self.pairs = all_pairs[:n_train]
        else:
            self.pairs = all_pairs[n_train:]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        x = torch.tensor([a, b], dtype=torch.long)
        y = torch.tensor((a + b) % self.p, dtype=torch.long)
        return x, y

# ==============================================================================
# UTILS
# ==============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    steps: int,
    noise_sigma: float = 0.0
) -> Dict:
    """Train model with optional noise injection."""
    model.train()
    
    # Noise injection hook
    def noise_hook(module, args, output):
        # output is (attn_output, attn_weights) tuple in MHA
        attn_out, attn_weights = output
        if noise_sigma > 0 and model.training:
            noise = torch.randn_like(attn_out) * noise_sigma
            return (attn_out + noise, attn_weights)
        return output

    handles = []
    if noise_sigma > 0:
        # Inject at Layer 0 (or all layers? Exp 4 used Layer 1. Let's use Layer 0 for deep impact)
        # DeepModularTransformer has 'blocks'
        # Let's inject at LAST layer to simulate "Decision Boundary" noise for hardening?
        # Or Early layer?
        # Exp A.2 (Hardening) injected at Layer 1 (pre-FFN).
        # Here we have 2 layers. Let's inject at Layer 0 (first layer) to harden downstream?
        # Or Layer 1?
        # Let's inject at ALL layers to be sufficiently hard.
        for block in model.blocks:
            handles.append(block.attn.register_forward_hook(noise_hook))
    
    print(f"Training for {steps} steps (σ={noise_sigma})...")
    
    step = 0
    patience_counter = 0
    consecutive_success_steps = 3
    
    # Infinite iterator logic
    while step < steps:
        for x, y in train_loader:
            if step >= steps:
                break
                
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            loss.backward()
            optimizer.step()
            
            # Logging & Early Stopping Check
            if step % 500 == 0:
                # Check Train Acc (batch)
                train_acc = (logits.argmax(dim=-1) == y).float().mean().item()
                
                # Check Generalization (Test Acc)
                model.eval()
                correct_val = 0
                total_val = 0
                with torch.no_grad():
                    for xt, yt in test_loader:
                        xt, yt = xt.to(device), yt.to(device)
                        # Re-apply noise hook if needed? 
                        # No, validation should be clean?
                        # Wait, for Hardened model, we want it to be robust?
                        # The user aid "As soon as test accuracy >= 0.999".
                        # For Hardened, "Test Accuracy" usually means CLEAN accuracy?
                        # Or Robust accuracy?
                        # Standard Grokking is about CLEAN generalization.
                        # Hardened model trains with noise, but we want it to solve the task.
                        # We should check CLEAN test acc first.
                        # If Hardened model trains with sigma=2.0...
                        # It should achieve high accuracy on CLEAN data too.
                        # Let's check pure clean accuracy for early stopping.
                        # (Hooks are active on model? Yes. But noise_sigma arg in train_model controls hooks?)
                        # noise_hook checks 'if noise_sigma > 0'.
                        # The hook is registered on the model.
                        # We must disable hook for CLEAN validation?
                        # The hook is registered in 'handles'.
                        # We can just set a flag? Or remove hooks temporarily?
                        # Easier: Just run validation. The hook adds noise if noise_sigma > 0.
                        # If we want CLEAN validation, we need to bypass the hook or set noise_sigma=0.
                        # But noise_sigma is fixed scope.
                        # Actuallly, for Hardened model, the noise is PART of the training buffer.
                        # "Grokking" for Hardened model means it learns to handle the noise?
                        # User said "Hardened Accuracy tops out ~0.95-0.98".
                        # So maybe 0.999 is too strict for Hardened?
                        # User said "Baseline... Test Acc >= 0.999".
                        # For "Hardened", maybe check robustness?
                        # But user said "Fix 1: Early-stop immediately after grokking... As soon as test accuracy >= 0.999".
                        # If Hardened never reaches 0.999 (due to noise), we might train forever.
                        # But Hardened usually reaches 1.0 on Train.
                        # Exp 5 logs showed Hardened reaching 1.0 acc/loss 0.08.
                        # So it CAN reach 1.0. 
                        # I will use 0.999 threshold for both.
                        # But I need to ensure the hook doesn't noise the validation pass if I want "Clean Test Acc".
                        # The hook is active.
                        # So validation will be noisy for Hardened model.
                        # If validation is noisy, 0.999 might be hard.
                        # Logs showed Hardened Acc 1.000 (Loss 0.08).
                        # So it IS possible.
                        # I'll stick to the logic.
                        
                        pt = model(xt).argmax(dim=-1)
                        correct_val += (pt == yt).sum().item()
                        total_val += yt.size(0)
                
                model.train()
                test_acc = correct_val / total_val
                
                print(f"  Step {step:5d} | Loss: {loss.item():.4f} | TrAcc: {train_acc:.3f} | TeAcc: {test_acc:.3f}")
                
                if test_acc >= 0.999:
                    patience_counter += 1
                    if patience_counter >= consecutive_success_steps:
                        print(">>> Early Stopping: Grokking Condition Met.")
                        # Remove hooks before returning (though hooks are local to handles list, 
                        # but we need to remove them to avoid pollution)
                        for h in handles:
                            h.remove()
                        return {'final_loss': loss.item(), 'status': 'grad_grok'}
                else:
                    patience_counter = 0
            
            step += 1
            
    # Cleanup hooks
    for h in handles:
        h.remove()
        
    return {'final_loss': loss.item()}

def evaluate_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    noise_levels: List[float] = [0.0, 2.0, 3.0, 5.0]
) -> Dict[float, float]:
    """Evaluate accuracy under varying noise levels."""
    model.eval()
    results = {}
    
    for sigma in noise_levels:
        # Register temporary hook
        handles = []
        if sigma > 0:
            def noise_hook(module, args, output):
                attn_out, attn_weights = output
                noise = torch.randn_like(attn_out) * sigma
                return (attn_out + noise, attn_weights)
                
            for block in model.blocks:
                handles.append(block.attn.register_forward_hook(noise_hook))
        
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        results[sigma] = acc
        
        # Cleanup
        for h in handles:
            h.remove()
            
    return results

# ==============================================================================
# EXPERIMENT
# ==============================================================================

def run_single_seed(
    p: int = 113,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    train_steps: int = 20000,
    hardening_sigma: float = 2.0,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1.0, # Grokking
    seed: int = 42,
    device: str = 'cuda'
) -> Dict:
    print(f"\n>>> SEED {seed}")
    setup_reproducibility(seed)
    
    # 1. Data
    # Use different seeds for train/test split consistency?
    # ModularAdditionDataset uses internal seed for shuffle.
    # We want consistent splits across runs if needed, but here we just need generalization.
    train_set = ModularAdditionDataset(p=p, split='train', train_frac=0.5, seed=seed)
    test_set = ModularAdditionDataset(p=p, split='test', train_frac=0.5, seed=seed)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size) # Full batch
    
    print(f"Dataset p={p}: Train {len(train_set)}, Test {len(test_set)}")
    
    # 2. Train BASELINE
    print(f"\nPhase 1: Training BASELINE (No Noise)")
    baseline = DeepModularTransformer(p=p, d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    opt_base = torch.optim.AdamW(baseline.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_model(baseline, train_loader, test_loader, opt_base, device, steps=train_steps, noise_sigma=0.0)
    
    base_res = evaluate_robustness(baseline, test_loader, device)
    print("Baseline Robustness:", base_res)
    
    if base_res[0.0] < 0.95:
        print("WARNING: Baseline failed to generalize!")
        return {'status': 'failed_generalization'}

    # 3. Train HARDENED
    print(f"\nPhase 2: Training HARDENED (σ={hardening_sigma})")
    hardened = DeepModularTransformer(p=p, d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    opt_hard = torch.optim.AdamW(hardened.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Train with noise
    train_model(hardened, train_loader, test_loader, opt_hard, device, steps=train_steps, noise_sigma=hardening_sigma)
    
    hard_res = evaluate_robustness(hardened, test_loader, device)
    print("Hardened Robustness:", hard_res)
    
    if hard_res[0.0] < 0.80:  # Hardened trains with noise, lower clean threshold expected
        print("WARNING: Hardened failed to generalize!")
        return {'status': 'failed_generalization'}
        
    # 4. Swap Geometry
    print(f"\nPhase 3: Creating HYBRID (Hardened G -> Baseline S)")
    hybrid = copy.deepcopy(baseline)
    
    # Swap QK
    swap_info = swap_qk_parameters(
        source_model=hardened,
        target_model=hybrid,
        layer_filter=list(range(n_layers)) # All layers
    )
    print(f"Swapped {swap_info['swapped_params']} parameters")
    
    # 5. Evaluate Hybrid
    hybrid_res = evaluate_robustness(hybrid, test_loader, device)
    print("Hybrid Robustness:", hybrid_res)
    
    # 6. Analysis
    # Check if Hybrid recovered robustness vs Baseline
    # Specifically at sigma=hardening_sigma
    base_acc = base_res[hardening_sigma]
    hard_acc = hard_res[hardening_sigma]
    hybr_acc = hybrid_res[hardening_sigma]
    
    # Recovery Ratio: (Hybrid - Base) / (Hard - Base)
    # If Base=0, Hard=100, Hybrid=0 -> Ratio 0.0 (Crash)
    # If Hybrid=100 -> Ratio 1.0 (Recovery)
    
    denom = max(hard_acc - base_acc, 1e-6)
    recovery_ratio = (hybr_acc - base_acc) / denom
    
    # Check residual alignment (Baseline vs Hybrid)
    # To see if G forced S into alignment or ignored it
    # We need to run forward pass on some data
    x_sample, _ = next(iter(test_loader))
    x_sample = x_sample[:100].to(device)
    
    with torch.no_grad():
        _ = baseline(x_sample)
        v_base = baseline.get_residual()
        _ = hardened(x_sample)
        v_hard = hardened.get_residual()
        _ = hybrid(x_sample)
        v_hybr = hybrid.get_residual()
        
    cossim_base_hybr = layer_cosine_similarity({'out': v_base}, {'out': v_hybr}, 'out')
    
    result = {
        'seed': seed,
        'baseline_robustness': base_res,
        'hardened_robustness': hard_res,
        'hybrid_robustness': hybrid_res,
        'recovery_ratio': recovery_ratio,
        'cossim_base_hybr': cossim_base_hybr,
        'status': 'success'
    }
    
    return result

def run_experiment(
    n_seeds: int = 3,
    start_seed: int = 42,
    device: str = 'cuda',
    **kwargs
):
    all_results = []
    
    for i in range(n_seeds):
        res = run_single_seed(seed=start_seed + i, device=device, **kwargs)
        all_results.append(res)
        
    # Aggregate
    successful = [r for r in all_results if r['status'] == 'success']
    if not successful:
        print("No successful runs.")
        return
        
    ratios = [r['recovery_ratio'] for r in successful]
    mean_ratio = np.mean(ratios)
    
    print("\n================================================================================")
    print("EXPERIMENT 5 AGGREGATE RESULTS (Modular Arithmetic)")
    print("================================================================================")
    print(f"Mean Recovery Ratio: {mean_ratio:.2f} ± {np.std(ratios):.2f}")
    
    outcome = "CRASH" if mean_ratio < 0.2 else "RECOVERY" if mean_ratio > 0.8 else "PARTIAL"
    print(f"Outcome: {outcome}")
    
    # Save
    save_path = Path("data/exp_5_geometry_recovery_results.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=113)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--train_steps', type=int, default=20000)
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    run_experiment(
        p=args.p,
        d_model=args.d_model,
        n_layers=args.n_layers,
        train_steps=args.train_steps,
        n_seeds=args.n_seeds,
        device=args.device
    )

if __name__ == '__main__':
    main()
