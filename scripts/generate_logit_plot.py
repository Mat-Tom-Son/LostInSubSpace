
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from clean_audit.experiments.exp_a_foundation import SimpleTransformer, ModularArithmeticDataset
from clean_audit.lib.clamps import NaiveClamp

# Local imports
sys.path.append(os.getcwd())

def generate_plot():
    print("Generating Signed Margin Plot...")
    
    # Config (Hardened Model)
    p = 113
    d_model = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Model
    model = SimpleTransformer(vocab_size=128, d_model=32, n_heads=4, max_seq_len=16, disable_ffn=False).to(device)
    model.load_state_dict(torch.load('hardened_q3_q4.pt', map_location=device))
    model.eval()
    
    # Dataset
    dataset = ModularArithmeticDataset(p=p, seq_len=16, train=False)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    # helper
    def get_margins(noise=0.0, clamp=None):
        margins = []
        # Setup model state
        model.post_clamp_noise_scale = noise
        model.clamp_fn = clamp
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                # Last token only
                logits = logits[:, -1, :]
                targets = y[:, -1]
                
                # Compute Signed True Margin
                # margin = logit[target] - max(logit[~target])
                
                # 1. Get target logits
                target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                
                # 2. Get max other
                # Mask out targets
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask.scatter_(1, targets.unsqueeze(1), False)
                other_logits = logits[mask].view(logits.size(0), -1)
                max_other = other_logits.max(dim=1).values
                
                margin = target_logits - max_other
                margins.append(margin.cpu().numpy())
                
                if len(margins) * 1024 > 5000: break
        
        return np.concatenate(margins)

    print("Collecting Margins...")
    # 1. Healthy (Hardened Clean)
    m_clean = get_margins(noise=0.0)
    
    # 2. Injured (Hardened + Noise 5.0)
    m_noise = get_margins(noise=5.0)
    
    # 3. Sedated (Hardened + Clamp 3.0 + Noise 5.0)
    # Note: Clamp is post-LN.
    clamp = NaiveClamp(3.0, layer_idx=-1)
    m_sedated = get_margins(noise=5.0, clamp=clamp)
    
    # Calculate Ratio (Bonus Check)
    # Ratio = m_sedated_clean / m_clean (without noise for clean comparison?)
    # User suggested comparing sedated/clean margins directly.
    # Let's get Clean Sedated (no noise) to check scaling law
    m_sedated_clean = get_margins(noise=0.0, clamp=clamp)
    ratio = m_sedated_clean / (m_clean + 1e-6) # avoid div zero
    avg_ratio = np.median(ratio)
    print(f"Margin Scaling Ratio (Sedated/Clean): {avg_ratio:.4f}")
    # Expected: 3.0 / 4.33 = 0.69
    
    # Plotting - Premium Style (Smoothed Density + Inset ECDF)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # helper for gaussian smoothing
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    # Common bins
    all_data = np.concatenate([m_clean, m_noise, m_sedated])
    min_v, max_v = all_data.min(), all_data.max()
    bins = np.linspace(min_v, max_v, 150) # more bins for smoothness
    
    def get_density(data, bins):
        y, x = np.histogram(data, bins=bins, density=True)
        centers = 0.5*(x[1:] + x[:-1])
        # Stronger smoothing
        y_smooth = smooth(y, 10) # smooth over 10 bins
        return centers, y_smooth

    # Generate Curves
    c_noise, y_noise = get_density(m_noise, bins)
    c_sed, y_sed = get_density(m_sedated, bins)
    
    # 1. Clean Reference (Line Only to save scale)
    clean_mean = np.mean(m_clean)
    ax.axvline(clean_mean, color='green', linestyle='--', linewidth=2, label=f'Clean Margin (~{clean_mean:.1f})')
    # Add a visual "anchor" for clean
    ax.text(clean_mean+0.5, ax.get_ylim()[1]*0.9, 'Clean Baseline', color='green', rotation=90, va='top', ha='left', fontsize=9)
    
    # 2. Plot Curves
    # Injured: Orange Curve
    ax.plot(c_noise, y_noise, color='orange', linewidth=2, label='Injured (Noise)')
    
    # Sedated: Red Curve
    ax.plot(c_sed, y_sed, color='#D32F2F', linewidth=2, label='Sedated (Clamp)')
    
    # 3. Highlight Failure Zones (Visceral)
    # Shade area < 0 for Sedated
    err_mask_sed = c_sed < 0
    ax.fill_between(c_sed, 0, y_sed, where=err_mask_sed, color='#D32F2F', alpha=0.3)
    
    # Shade area < 0 for Injured
    err_mask_noise = c_noise < 0
    if err_mask_noise.any():
        ax.fill_between(c_noise, 0, y_noise, where=err_mask_noise, color='orange', alpha=0.3)
    
    # Labels (Smart Positioning)
    # Red Label: Top Left of the red mass
    fail_sed = (m_sedated < 0).mean() * 100
    # Find peak in negative region
    neg_indices = np.where(c_sed < 0)[0]
    if len(neg_indices) > 0:
        peak_idx = neg_indices[np.argmax(y_sed[neg_indices])]
        ax.annotate(f'{fail_sed:.1f}% Fail', 
                    xy=(c_sed[peak_idx], y_sed[peak_idx]), 
                    xytext=(-10, y_sed[peak_idx]*1.5),
                    arrowprops=dict(facecolor='#D32F2F', shrink=0.05),
                    color='#D32F2F', fontweight='bold')
    
    # Orange Label: Lower down or Top Right of orange mass?
    # Orange mass crosses 0 but mean is pos. 
    # Let's put label right at x=-1 if density exists?
    fail_noise = (m_noise < 0).mean() * 100
    if fail_noise > 1.0:
        # Just text annotation near x=0
        ax.annotate(f'{fail_noise:.1f}%', 
                    xy=(0, y_noise[np.searchsorted(c_noise, 0)]), 
                    xytext=(-5, y_noise[np.searchsorted(c_noise, 0)] + 0.05),
                    arrowprops=dict(facecolor='orange', shrink=0.05),
                    color='darkorange', fontweight='bold')

    # 4. Decision Boundary
    ax.axvline(0, color='black', linewidth=2, alpha=0.8)
    
    # 5. Inset ECDF
    ins = ax.inset_axes([0.05, 0.55, 0.30, 0.30]) # Moved slightly left
    
    def plot_ecdf(data, color, style='-'):
        x = np.sort(data)
        y = np.linspace(0, 1, len(data))
        ins.plot(x, y, color=color, linestyle=style, linewidth=1.5)
        fail_r = (data < 0).mean()
        if fail_r > 0.01:
            ins.plot(0, fail_r, 'o', color=color, markersize=4)
            
    plot_ecdf(m_clean, 'green', '--')
    plot_ecdf(m_noise, 'orange')
    plot_ecdf(m_sedated, '#D32F2F')
    
    ins.axvline(0, color='black', linewidth=1, alpha=0.5)
    ins.set_title('Risk (ECDF)', fontsize=8)
    ins.set_xlim(-20, 20)
    ins.set_ylim(0, 1.05)
    ins.grid(True, alpha=0.3)
    ins.tick_params(labelsize=7)

    # Final Polish
    ax.set_xlabel('Signed True Margin (Signal - Noise)')
    ax.set_ylabel('Density')
    ax.set_title('Sedation shrinks true-margin until noise crosses the decision boundary', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    # Auto-scale Y to fit curves (ignore Clean line height)
    ax.set_ylim(0, max(y_noise.max(), y_sed.max()) * 1.2)
    ax.set_xlim(min_v, max(clean_mean * 1.2, max_v))
    
    plt.savefig('logit_gaps.png', dpi=300)
    print("Saved logit_gaps.png")

if __name__ == "__main__":
    generate_plot()
